import train
from train import load_best_model, calculate_bleu, load_data, calculate_val_loss
from pathlib import Path
import sentencepiece
from torchmetrics.text import SacreBLEUScore
import torch
import json

class NoOp:
  def __call__(self, *args, **kwargs):
    return self

  def __getattr__(self, name):
    return self

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    return False  # Don't suppress real exceptions

directory = Path("./runs/")
values = {}

for entry in directory.iterdir():
    if entry.is_dir() and entry.name != "wmt14":
        train.save_file_path_root = entry
        train.config_path = entry / "config.json"
        train.writer = NoOp()

        model,loss_fn = load_best_model()

        train.model = model
        train.loss_fn = loss_fn

        train.metrics = [
           torch.nn.CrossEntropyLoss(
                ignore_index=train.config["pad_token"]
            )
        ]

        sp_bpe = sentencepiece.SentencePieceProcessor()
        sp_bpe.load(f"tokenizers/{train.config["tokenizer_source"]}/bpe.model")
        sacreblue_metric = SacreBLEUScore(lowercase=True)
        
        val_loader = load_data(sp_bpe,load_train=False)

        train.val_loader = val_loader

        bleu = calculate_bleu(val_loader,model,sp_bpe).item() * 100

        val_loss,metrics = calculate_val_loss(False,torch.zeros((),device=train.DEVICE))

        print("\n====")
        print(entry.name)
        print(bleu)
        print(val_loss)
        print(metrics[0])
        values[entry.name] = {"bleu":bleu,"val_loss":val_loss,"non_smoothed":metrics[0]}

results_path = Path(__file__).resolve().parent / "artifacts" / "experiment_metrics.json"
results_path.parent.mkdir(parents=True, exist_ok=True)
with results_path.open("w") as f:
   json.dump(values,f,indent=2)
