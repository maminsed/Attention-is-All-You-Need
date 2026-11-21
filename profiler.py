import train
from train import calculate_bleu, load_best_model, load_data, serialize_config
from torch.profiler import profile, ProfilerActivity, schedule, record_function
import sentencepiece as spm
from pathlib import Path
from collections import defaultdict
import json

TRACE_FOLDER="./profiler_trace/nov_20/"

profiler_config={
    "mode": "calculate_bleu",
    "skip_first":11,
    "wait":4,
    "warmup":1,
    "active":2,

    "model_root": train.save_file_path_root,
    "note": "first run",
}

Path(TRACE_FOLDER).mkdir(parents=True,exist_ok=True)

totals = defaultdict(lambda: {"cpu_us":0,"cuda_us":0,"calls":0})
target_stages = {"prefill","decode","enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__"}

def trace_handler(p):
    step_num = int(p.step_num)
    print(f"\n\n====step: {step_num} returned.")

    for event in p.key_averages():
        if event.key in target_stages:
            totals[event.key]["cpu_us"]+=event.cpu_time_total
            totals[event.key]["cuda_us"]+=event.cuda_time_total
            totals[event.key]["calls"]+= event.count

    if step_num == 18 or step_num == 46: # first and last only.
        print(p.key_averages().table(sort_by="cuda_time_total",row_limit=10))
        p.export_chrome_trace(f"{TRACE_FOLDER}/trace_{step_num}.json")


if __name__ == "__main__":
    # Loading model
    model, _ = load_best_model()

    # Loading tokenizer
    sp_bpe = spm.SentencePieceProcessor()
    sp_bpe.load(f"tokenizers/{train.config["tokenizer_source"]}/bpe.model")

    # Loading data
    train_loader, val_loader, train_token_count = load_data(sp_bpe)
    print(f"Train Token Count: {train_token_count:,}")
    del train_loader

    # Storing config
    with open(f"{TRACE_FOLDER}/config.json","w") as f:
        json.dump(
            {
                "model_config": serialize_config(train.config),
                "profiler_config": profiler_config,
            }
            ,f,
            indent=2
        )

    # Num steps ~= 46
    with profile(
        activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
        schedule=schedule(
            skip_first=profiler_config["skip_first"],
            wait=profiler_config["wait"],
            warmup=profiler_config["warmup"], 
            active=profiler_config["active"]
        ), # 5 x 7
            on_trace_ready=trace_handler
    ) as prof:
        with record_function("calculate_bleu"):
            calculate_bleu(val_loader,model,sp_bpe,return_translations=False,step_fn=prof.step) 

    for stage, stats in totals.items():
        cpu_ms = stats["cpu_us"] / 1000.0
        cuda_ms = stats["cuda_us"] / 1000.0
        print(f"{stage} ({stats['calls']} calls) -> CPU: {cpu_ms:.2f} ms | GPU: {cuda_ms:.2f} ms")

    