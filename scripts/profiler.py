from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

import train
from train import calculate_bleu, load_best_model, load_data, serialize_config
from torch.profiler import profile, ProfilerActivity, schedule, record_function
import sentencepiece as spm
import torch
from collections import defaultdict
from types import SimpleNamespace
import json
import time
from contextlib import contextmanager

profiler_config={
    "mode": "calculate_bleu",
    "skip_first":11,
    "wait":4,
    "warmup":1,
    "active":2,

    "wrapper": "time", # profiler or time

    "model_root": train.save_file_path_root,
    "note": "first run",

    # calculate_bleu flags
    "kv_cache": True,
    "seperate_prefill": True,
    "precompute_pos_encs": True,
    "use_amp": True
}
if profiler_config["kv_cache"] == True and profiler_config["seperate_prefill"] == False:
    raise ValueError("kv cache requires seperate prefill")

totals = defaultdict(lambda: {"cpu_us":0,"cuda_us":0,"calls":0})
target_stages = {"prefill","decode","enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__"}

def trace_handler(p):
    step_num = int(p.step_num)
    print(f"\n\n====step: {step_num} returned.")

    for event in p.key_averages():
        if event.key in target_stages:
            totals[event.key]["cpu_us"]+=event.cpu_time_total
            totals[event.key]["cuda_us"]+=event.device_time_total
            totals[event.key]["calls"]+= event.count

    if step_num == 18 or step_num == 46: # first and last only.
        print(p.key_averages().table(sort_by="cuda_time_total",row_limit=10))
        p.export_chrome_trace(f"{TRACE_FOLDER}/trace_{step_num}.json")

@contextmanager
def time_counter(warmup_steps=0,total_steps=47):
    counter = SimpleNamespace(steps=0,start=None,total=0)

    def step():
        counter.steps+=1
        if counter.steps == warmup_steps:
            torch.cuda.synchronize()
            counter.start = time.perf_counter()
        if counter.steps == total_steps:
            torch.cuda.synchronize()
            counter.total = time.perf_counter()-counter.start
        

    counter.step = step
    
    if warmup_steps == 0:
        torch.cuda.synchronize()
        counter.start = time.perf_counter()
    

    try:
        yield counter
    finally:
        if counter.total == 0:
            torch.cuda.synchronize()
            counter.total = time.perf_counter()-counter.start
        print(f"total time elapsed: {counter.total:.4f}s")

def main():
    # Loading model
    model, _ = load_best_model()

    if profiler_config.get("precompute_pos_encs",False) is False:
        model.pos_encs = None

    # Loading tokenizer
    sp_bpe = spm.SentencePieceProcessor()
    sp_bpe.load(f"tokenizers/{train.config["tokenizer_source"]}/bpe.model")

    # Loading data
    train_loader, val_loader, train_token_count = load_data(sp_bpe)
    print(f"Train Token Count: {train_token_count:,}")
    del train_loader

    profiler_mode = profiler_config["wrapper"] == "profiler"

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
    ) if profiler_mode else time_counter(profiler_config["skip_first"],len(val_loader)) as prof:
        with record_function("calculate_bleu"):
            calculate_bleu(
                val_loader,
                model,
                sp_bpe,
                return_translations=False,
                step_fn=prof.step,
                profiler_config=profiler_config
            )

    if profiler_mode:
        for stage, stats in totals.items():
            cpu_ms = stats["cpu_us"] / 1000.0
            cuda_ms = stats["cuda_us"] / 1000.0
            print(f"{stage} ({stats['calls']} calls) -> CPU: {cpu_ms:.2f} ms | GPU: {cuda_ms:.2f} ms")
        result = totals
    else:
        result = {"total_seconds": prof.total,"total_steps": prof.steps}

    # Storing config
    with open(f"{TRACE_FOLDER}/config.json","w") as f:
        json.dump(
            {
                "model_config": serialize_config(train.config),
                "profiler_config": profiler_config,
                "target_stages": result,
            }
            ,f,
            indent=2
        )

def construct_note():
    return f"{profiler_config["wrapper"]}_{"kv" if profiler_config["kv_cache"] else "no_kv"}_{"seperate" if profiler_config["seperate_prefill"] else "combined"}_{"precomputedEncs" if profiler_config["precompute_pos_encs"] else "recomputedEncs"}_{"amp" if profiler_config["use_amp"] else "noAmp"}"
    

if __name__ == "__main__":
    MODE="all" # single or all
    if MODE == "all":
        print("running in all mode")
        for kv_cache,seperate_prefill,pos_encs,use_amp in ((False,False,False,False),(True,True,True,False),(True,True,True,True)):
            print("\n\n\n\n===============")
            profiler_config["kv_cache"] = kv_cache
            profiler_config["seperate_prefill"] = seperate_prefill
            profiler_config["precompute_pos_encs"] = pos_encs
            profiler_config["use_amp"] = use_amp

            profiler_config["note"] = construct_note()
            
            TRACE_FOLDER=f"./artifacts/profiler_trace/{profiler_config["note"]}"
            Path(TRACE_FOLDER).mkdir(parents=True,exist_ok=True)

            main()

            totals.clear()
    else:
        print("running in single mode")
        profiler_config["note"] = construct_note()
        TRACE_FOLDER=f"./artifacts/profiler_trace/{profiler_config["note"]}"
        Path(TRACE_FOLDER).mkdir(parents=True,exist_ok=True)
        main()
