import json
import math
import random
from datetime import datetime
import time
from os import path
import os
import re
import sys

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from frozendict import frozendict
from torch.nn import ModuleList
from torch.nn.utils.rnn import pad_sequence

# from torchinfo import summary
from torch.utils.data import DataLoader, Dataset

PAD_TOKEN = 3
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CONTEXT_CAP = 128
save_file_path_root = "./test/overnight_Sep_19_v2"
history_path = f"{save_file_path_root}/history.json"
optimizer_path = f"{save_file_path_root}/optimizer.pt"
scheduler_path = f"{save_file_path_root}/scheduler.pt"

LOAD_PRE_SAVED = False
VERBOSE = True
VERBOSE_FREQ = 200
NUM_BATCHES = None
STOP_AFTER_EPOCH=20
NUM_TOKENS=0

if not path.exists(save_file_path_root):
    os.mkdir(save_file_path_root)

if not path.exists(f"{save_file_path_root}/best/"):
    os.mkdir(f"{save_file_path_root}/best/")

config = {
    # model params
    "vocab_size": 7000,
    "d_model": 512,
    "d_v": 64,
    "d_k": 64,
    "d_ff": 2048,  # MLPs hidden layers activation,
    "h": 8,  # number of heads per attention layer, h * d_k = d_model
    "num_enc_layers": 6,  # number of encoder layers
    "num_dec_layers": 6,  # number of encoder layers
    "MLP_activation": "relu",
    ## Regularization/activations
    "drop_pre_norm": 0.2,
    "drop_post_emb": 0.2,

    # Training params
    "benchmark_mode": False,
    "batch_size": 64,
    "epochs": 6,  # just for initial testing
    "early_stopping_limit": 5,
    ## Optimizer
    "optimizer": "adam",  # informational only,
    "lr": 1.0, # scheduler will take care
    "optim_betas": [0.9, 0.98],
    "optim_eps": 1e-9,
    ### Scheduler
    "warmup_steps": 1000, # lowering a bit since were training for less epochs
    ## Loss
    "loss": "CrossEntropyLoss",  # informational only
    "label_smoothing": 0.1,

    # Dataset params
    "context_size": CONTEXT_CAP,
    "pad_token": PAD_TOKEN,
    "test_size": 0.2,
    "val_size": 0.2,
    "title_use_in_tokenization": True,
    "mega_batch_size": 100, 

    # Device params
    "device": DEVICE,
    "dtype": torch.float32,
    "seed": 42,
    ## Multiproccessing
    "use_amp": False,
    "autocast_dtype": torch.bfloat16 if DEVICE=="mps" else torch.float16,
    "num_workers": 2, # will time. 
    "pin_memory": False,  # doesn't help much in mac and non_blocking weird.
}
assert config["h"] * config["d_v"] == config["d_model"]
assert DEVICE == "mps"

random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])

# generally useful param:
SQRT_INV_DM = 1/math.sqrt(config["d_model"])

# make all temp changes here to not mess up anything else and forget:
if config["benchmark_mode"]:
    config["epochs"] = 4 
    config["warmup_steps"] = 12

    config["num_workers"] = 4
    config["use_amp"] = False
    VERBOSE_FREQ=5
    VERBOSE=True


class AttnDataset(Dataset):
    def __init__(self, X: pd.Series, y: pd.Series):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ShuffledNearSampler:
    def __init__(self, data, batch_size,lengths):
        super().__init__()
        self.data:AttnDataset = data
        self.batch_size = batch_size
        self.lengths = lengths

    def __iter__(self):
        size = len(self.data)
        meg_size = self.batch_size * config["mega_batch_size"]

        indecies = np.random.permutation(size)
        megabatches = [sorted(indecies[i:i+meg_size],key=lambda x :self.lengths[x]) for i in range(0,size,meg_size)]

        megabatches = [[idx[i:i+self.batch_size] for i in range(0,len(idx),self.batch_size)] for idx in megabatches]
        megabatches = [i for megabatch in megabatches for i in megabatch]

        mb_indecies = np.random.permutation(len(megabatches))
        for mbi in mb_indecies:
            yield megabatches[mbi]

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)


def collate_fn(batch):
    src, tgt = zip(*batch)

    src = [torch.tensor(item) for item in src]  # np type carries over
    tgt = [torch.tensor(item) for item in tgt]

    src = pad_sequence(src, batch_first=True, padding_value=PAD_TOKEN)

    tgt = pad_sequence(tgt, batch_first=True, padding_value=PAD_TOKEN)

    return src, tgt



def load_data() -> tuple[DataLoader, DataLoader]:
    sp_bpe = spm.SentencePieceProcessor()
    sp_bpe.load("bpe.model")
    def encode(item):
        global NUM_TOKENS
        ids = sp_bpe.encode_as_ids(item, add_bos=True)
        NUM_TOKENS+=len(ids)
        return np.array(ids[:CONTEXT_CAP-1]+[sp_bpe.eos_id()], dtype=np.int64)

    train_df = pd.read_csv("train_df.csv",nrows=1000 if config["benchmark_mode"] else None)
    val_df = pd.read_csv("val_df.csv")
    title_df = pd.read_csv("title_df.csv")

    print(f"{datetime.now()}: # train sentences {train_df.shape[0]:,}")
    print(f"{datetime.now()}: # train docs {title_df.shape[0]:,}")
    print(f"{datetime.now()}: # val sentences {val_df.shape[0]:,}")
    print(train_df.head())

    X_train = [encode(item) for item in train_df["detext"]]
    y_train = [encode(item) for item in train_df["entext"]]

    X_val = [encode(item) for item in val_df["detext"]]
    y_val = [encode(item) for item in val_df["entext"]]
    print(f"{datetime.now()}: encoding done! Total number of tokens: {NUM_TOKENS:,}")

    print(f"{datetime.now()}: \n\n==============\nencode_as_ids:")
    print(
        sp_bpe.encode_as_ids(train_df.loc[0, "entext"][:10], add_bos=True, add_eos=True)
    )

    global NUM_BATCHES
    NUM_BATCHES = math.ceil(train_df.shape[0] / config["batch_size"])
    print(f"{datetime.now()}: batches/epoch: {NUM_BATCHES:,}")
    if NUM_BATCHES * config["epochs"] * 0.5 < config["warmup_steps"]:
        print(f"{datetime.now()}: WARNING: more than half of epochs are warmup steps. ")

    # setting up the data

    train_set = AttnDataset(X_train, y_train)
    val_set = AttnDataset(X_val, y_val)
    # test_set = AttnDataset(X_test,y_test)

    train_lengths=[len(x) for x in X_train]
    bs = ShuffledNearSampler(train_set,config["batch_size"],lengths=train_lengths)
    train_loader = DataLoader(
        train_set,
        # batch_size=config["batch_size"],
        # shuffle=True,
        batch_sampler=bs,
        collate_fn=collate_fn,
        pin_memory=config["pin_memory"],
        num_workers=config["num_workers"],
        persistent_workers=bool(config["num_workers"]),
    )

    val_loader = DataLoader(
        val_set, batch_size=config["batch_size"], collate_fn=collate_fn, pin_memory=config["pin_memory"]
    )
    return train_loader, val_loader


def activation_lookup(activation="relu"):
    # will add more activations if required
    return frozendict(
        {
            "relu": torch.nn.ReLU(),
        }
    )[activation]


class SelfAttention(torch.nn.Module):
    def __init__(self, config, *, lookahead_mask=False, pad_mask=True):
        super().__init__()

        self.config = config
        self.lookahead_mask = lookahead_mask
        self.pad_mask = pad_mask
        self.d_k_sqrt = math.sqrt(self.config["d_k"])

        self.W_K = torch.nn.Linear(config["d_model"], config["d_k"]*config["h"], bias=False) # CUDA
        self.W_Q = torch.nn.Linear(config["d_model"], config["d_k"]*config["h"], bias=False) # CUDA
        self.W_V = torch.nn.Linear(config["d_model"], config["d_v"]*config["h"], bias=False) # CUDA
        
        self.W_O = torch.nn.Linear(
            config["h"] * config["d_v"], config["d_model"], bias=False
        )  # CUDA
        self.Dropout = torch.nn.Dropout(config["drop_pre_norm"])
        self.LayerNorm = torch.nn.LayerNorm(config["d_model"])

    def apply_mask(self, x, mask_constant):
        return x.where(~mask_constant, float("-inf"))

    def forward(self, x, y, padding_mask_constant=None, lookahead_mask_constant=None):

        # x: B,S,d_m
        # y: B,Sy,d_m
        # (in cases of self attention, S=Sy)
        B,S,dm = x.shape
        Sy = y.shape[1]

        K = self.W_K(x) # CUDA-B,S,dk*H
        V = self.W_V(x) # CUDA-B,S,dv*H
        Q = self.W_Q(y) # CUDA-B,Sy,dk*H

        K = K.view((B,S ,self.config["h"],-1)).transpose(1,2) # CUDA-B,H,S,dk
        Q = Q.view((B,Sy,self.config["h"],-1)).transpose(1,2) # CUDA-B,H,Sy,dk
        V = V.view((B,S ,self.config["h"],-1)).transpose(1,2) # CUDA-B,H,S,dv

        y_2 = Q @ K.transpose(2,3) / self.d_k_sqrt # CUDA-B,H,Sy,S
        if self.lookahead_mask:
            y_2 = self.apply_mask(y_2, lookahead_mask_constant) # ***

        if self.pad_mask:
            y_2 = self.apply_mask(y_2, padding_mask_constant)  # ***

        y_2 = torch.softmax(y_2,dim=3) @ V # CUDA-B,H,Sy,dv
        y_2 = y_2.transpose(1,2).reshape((B,Sy,dm)) # CUDA-B,Sy,dm

        y_2 = self.W_O(y_2)
        y_2 = self.Dropout(y_2)
        y = self.LayerNorm(y+y_2)
        return y # CUDA-B,Sy,dm


class MLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.layer1 = torch.nn.Linear(config["d_model"], config["d_ff"])  # CUDA
        self.activation = activation_lookup(config["MLP_activation"])
        self.layer2 = torch.nn.Linear(config["d_ff"], config["d_model"])  # CUDA

        self.Dropout = torch.nn.Dropout(config["drop_pre_norm"])
        self.LayerNorm = torch.nn.LayerNorm(config["d_model"])

    def forward(self, x):
        x_2 = x  # CUDA- B,S,dm
        x = self.layer1(x)  # CUDA-B,S,d_ff
        x = self.activation(x)  # CUDA- B,S,d_ff
        x = self.layer2(x)  # CUDA- B,S,dm
        x = self.Dropout(x)  # CUDA- B,S,dm
        x = self.LayerNorm(x + x_2)  # CUDA - B,S,dm
        return x


class Transformer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.pad_token = self.config["pad_token"]
        self.embedding = torch.nn.Embedding(config["vocab_size"], config["d_model"])
        torch.nn.init.uniform_(self.embedding.weight,-SQRT_INV_DM,SQRT_INV_DM)
        self.dropout = torch.nn.Dropout(config["drop_post_emb"])

        # encoder
        self.encoder_attention = ModuleList(
            [SelfAttention(config) for _ in range(config["num_enc_layers"])]
        )
        self.encoder_MLP = ModuleList(
            [MLP(config) for _ in range(config["num_enc_layers"])]
        )

        # decoder
        self.decoder_attention = ModuleList(
            [
                SelfAttention(config, lookahead_mask=True, pad_mask=False)
                for _ in range(config["num_dec_layers"])
            ]
        )
        self.cross_attention = ModuleList(
            [SelfAttention(config) for _ in range(config["num_dec_layers"])]
        )
        self.decoder_MLP = ModuleList(
            [MLP(config) for _ in range(config["num_dec_layers"])]
        )

    def positional_encoding(self, seq_len, device=None, d_model=None, n=10000):
        d_model = self.config["d_model"] if d_model is None else d_model
        pe = torch.zeros(seq_len, d_model, device=device)  # CUDA- S,d_m
        pos = torch.arange(seq_len, device=device).reshape(-1, 1)  # CUDA- S,1
        freq = torch.arange(0, d_model, 2, device=device)  # CUDA, d_m/2
        dim = n ** (freq / d_model)  # CUDA, d_m/2

        pe[:, 0::2] = torch.sin(pos / dim)
        pe[:, 1::2] = torch.cos(pos / dim)
        return pe  # CUDA- S,d_m

    def forward(self, x, y):
        # the device names are written next to it assuming that config["device"] is CUDA and the rest is the shape. we assume input shape is:
        # x.shape = CUDA- B,S
        # y.shape = CUDA- B,Sy
        # returns: CUDA- B,Sy,VS

        # masking constants:
        d = y.shape[1]
        lookahead_mask_constant = torch.triu(
            torch.ones((d, d), device=y.device, requires_grad=False), 1
        ).bool()  # CUDA - Sy,Sy

        padding_mask_constant = x == self.pad_token  # CUDA- B,S
        padding_mask_constant.requires_grad = False
        padding_mask_constant = padding_mask_constant.reshape(
            (x.shape[0], 1, 1, x.shape[1])
        )  # CUDA- B,1,1,S

        # x embedding
        x = self.embedding(x)  # CUDA- B,S,dm
        x = x / SQRT_INV_DM  # CUDA- B,S,d_m (INV_INV_SQRT: SQRT_DM)
        # positional encoding
        pe = self.positional_encoding(x.shape[1], device=x.device)  # CUDA- S,d_m
        x = x + pe  # CUDA- B,S,d_m
        x = self.dropout(x)

        # y embedding
        y = self.embedding(y)  # CUDA- B,Sy,dm
        y = y / SQRT_INV_DM  # CUDA- B,Sy,d_m
        pe = self.positional_encoding(y.shape[1], device=y.device)  # CUDA- Sy,d_m
        y = y + pe  # CUDA- B,Sy,d_m
        y = self.dropout(y)

        # for each layer
        for i in range(self.config["num_enc_layers"]):
            # encoder
            x = self.encoder_attention[i](x, x, padding_mask_constant)  # CUDA- B,S,d_m
            x = self.encoder_MLP[i](x)  # CUDA, B,S,dm

        for i in range(self.config["num_dec_layers"]):
            # decoder
            y = self.decoder_attention[i](
                y, y, lookahead_mask_constant=lookahead_mask_constant
            )  # CUDA- B,Sy,dm
            y = self.cross_attention[i](x, y, padding_mask_constant)  # CUDA- B,Sy,dm
            y = self.decoder_MLP[i](y)  # CUDA- B,Sy,dm

        # unembedding
        y = y @ self.embedding.weight.T  # CUDA - B,Sy,VS
        return y  # CUDA- B,Sy,VS



def calculate_val_loss():
    # val loss:
    model.eval()
    avg_vloss = torch.zeros((),device=DEVICE)
    avg_metrics = torch.zeros((len(metrics),),device=DEVICE)
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device=DEVICE, non_blocking=config["pin_memory"])
            y = y.to(device=DEVICE, non_blocking=config["pin_memory"])

            with torch.autocast(
                device_type=DEVICE, dtype=config["autocast_dtype"], enabled=config["use_amp"]
            ):
                # switching to torch.bfloat16 on mac due to not full support on mac
                logits_val = model(x, y[:, :-1])

                logits_val = logits_val.reshape((-1, logits_val.shape[-1]))
                y = y[:, 1:].reshape(-1)

                loss_val = loss_fn(logits_val, y)
                for i,m in enumerate(metrics):
                    avg_metrics[i] += m(logits_val,y).detach()

            
            avg_vloss += loss_val.detach()
    
    avg_vloss /= len(val_loader)
    avg_metrics /= len(val_loader)

    return avg_vloss.item(), [m.item() for m in avg_metrics]

def save_hist(history_path,model_path=None,optim_path=None,scheduler_path=None,*,save_weights=True,is_best=True):
    if model_path and optim_path and save_weights:
        torch.save(model.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optim_path)
        torch.save(scheduler.state_dict(),scheduler_path)

        if is_best:
            model_best_path = f"{save_file_path_root}/best/model.pt"
            optim_best_path = f"{save_file_path_root}/best/optimizer.pt"
            scheduler_best_path = f"{save_file_path_root}/best/scheduler.pt"

            torch.save(model.state_dict(), model_best_path)
            torch.save(optimizer.state_dict(), optim_best_path)
            torch.save(scheduler.state_dict(), scheduler_best_path)
        
    with open(history_path, "w") as f:
        json.dump(
            {
                "history": history,
                "config": {k:str(v) for k,v in config.items()},
                "last_epoch_vloss": last_epoch_vloss,
                "non_improve_cnt": non_improve_cnt,
                "best_val_loss": best_val_loss,
                "epoch": epoch,
                "model_path": model_path,
                "history_path": history_path,
                "optim_path": optim_path,
                "scheduler_path": scheduler_path
            },
            f,
        )



def train_one_epoch(epoch):
    print(f"{datetime.now()}: Epoch: {epoch}")

    total_loss = torch.zeros((),device=DEVICE)
    avg_metrics = torch.zeros(((len(metrics),)),device=DEVICE)
    total_time = 0
    total_batches = len(train_loader)

    epoch_start = time.time()
    for i, (x, y) in enumerate(train_loader):
        # x: B,S
        # y: B,Sy
        start_time = time.time()
        x = x.to(device=DEVICE, non_blocking=config["pin_memory"])
        y = y.to(device=DEVICE, non_blocking=config["pin_memory"])
        optimizer.zero_grad(set_to_none=True)  # default behaviour
        avg_metric = torch.zeros(((len(metrics),)),device=DEVICE)

        with torch.autocast(
            device_type=DEVICE, dtype=config["autocast_dtype"], enabled=config["use_amp"]
        ):
            logits = model(x, y[:, :-1])  # CUDA - B,Sy,VS

            logits = logits.reshape((-1,logits.shape[-1])) # (B*Sy,VS)
            y = y[:,1:].reshape(-1) # (B*Sy,)

            loss_value = loss_fn(logits, y)  # input, target

            with torch.no_grad():
                for j,m in enumerate(metrics):
                    avg_metric[j]+=m(logits,y)

        scaler.scale(loss_value).backward()
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
        if config["benchmark_mode"]:
            torch.mps.synchronize()
        end_time = time.time()

        # loss.backward()
        # optimizer.step()

        total_loss += loss_value.detach()
        avg_metrics += avg_metric.detach()
        delta = end_time - start_time
        total_time+=delta

        if i % VERBOSE_FREQ == 0:
            ts = (x.shape[0]*x.shape[1]+y.shape[0]) / delta
            history.append(
                {
                    "batch_loss": loss_value.item(),
                    "batch_percentage": i / NUM_BATCHES,
                    "batch_idx": i,
                    "epoch": epoch,
                    "batch_time": delta,
                    "tokens/sec": ts,
                }
            )
            if VERBOSE:
                print(
                    f"{datetime.now()}: [epoch {epoch}] batch {i} loss_value: {loss_value:.4f}, batch_time (s): {delta:,.3f}, tokens/sec: {ts:,.3f}"
                )
    epoch_end = time.time()
    avg_vloss, avg_val_metrics = calculate_val_loss()
    return (
        (total_loss / total_batches).item(), 
        [m.item() for m in (avg_metrics / total_batches)], 
        avg_vloss,
        avg_val_metrics, 
        total_time/total_batches,
        epoch_end-epoch_start
    )

def lr_update_fn(step):
    step = max(step,1)
    lr = SQRT_INV_DM*min(step**-0.5,step*config["warmup_steps"]**-1.5)
    return lr

def load_best_model():
    model = Transformer(config).to(config["device"], dtype=config["dtype"])
    print("\n\n==============")
    num_parameters=sum(p.numel() for p in model.parameters())
    non_embd_parameters=sum(p.numel() for n,p in model.named_parameters() if n != 'embedding.weight')
    print(f"num_parameters: {num_parameters:,} - num_non_embedding_params: {non_embd_parameters:,}")


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        betas=config["optim_betas"],
        eps=config["optim_eps"],
    )
    print(f"{datetime.now()}: optimizer loaded")

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_update_fn
    )
    print(f"{datetime.now()}: scheduler loaded")

    # B,C,d_1,..,d_n expected.
    loss_fn = torch.nn.CrossEntropyLoss(
        label_smoothing=config["label_smoothing"], ignore_index=config["pad_token"]
    )
    print(f"{datetime.now()}: loss_fn loaded")

    metrics = []
    non_smoothed_loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=config["pad_token"]
    )
    non_smoothed_loss_fn.name = "non_smoothed_loss_fn"
    metrics.append(non_smoothed_loss_fn)
    print(f"{datetime.now()}: metrics loaded")

    history = []
    last_epoch_vloss = float("inf")
    non_improve_cnt = 0
    best_val_loss = float("inf")
    start_epoch = -1
    if LOAD_PRE_SAVED:
        print(f"{datetime.now()}: loading best pre saved")
        assert path.exists(history_path)
        assert path.exists(optimizer_path)
        with open(history_path, "r") as f:
            dic = json.load(f)
            history = dic["history"]
            last_epoch_vloss = dic["last_epoch_vloss"]
            non_improve_cnt = dic["non_improve_cnt"]
            best_val_loss = dic["best_val_loss"]
            start_epoch = dic["epoch"]
            model_path = dic["model_path"]
        assert path.exists(model_path)

        model_best_path = f"{save_file_path_root}/best/model.pt"
        optimizer_best_path = f"{save_file_path_root}/best/optimizer.pt"
        scheduler_best_path = f"{save_file_path_root}/best/scheduler.pt"

        if path.exists(model_best_path):
            print("best model path found.")
            optimizer.load_state_dict(torch.load(optimizer_best_path))
            scheduler.load_state_dict(torch.load(scheduler_best_path))
            model.load_state_dict(torch.load(model_best_path))
        else:
            optimizer.load_state_dict(torch.load(optimizer_path))
            scheduler.load_state_dict(torch.load(scheduler_path))
            model.load_state_dict(torch.load(model_path))


        model.to(
            config["device"], dtype=config["dtype"]
        )

        
        print(
            f"{datetime.now()}: Starting from pre saved with: start_epoch: {start_epoch}, best_val_loss: {best_val_loss:.4f}, last_epoch_vloss: {last_epoch_vloss:.2f}, non_improve_cnt: {non_improve_cnt:.2f}"
        )
    return model,optimizer,scheduler,loss_fn,metrics,history,last_epoch_vloss,non_improve_cnt,best_val_loss,start_epoch


def load_model():
    model = Transformer(config).to(config["device"], dtype=config["dtype"])
    print("\n\n==============")
    num_parameters=sum(p.numel() for p in model.parameters())
    non_embd_parameters=sum(p.numel() for n,p in model.named_parameters() if n != 'embedding.weight')
    print(f"num_parameters: {num_parameters:,} - num_non_embedding_params: {non_embd_parameters:,}")


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        betas=config["optim_betas"],
        eps=config["optim_eps"],
    )
    print(f"{datetime.now()}: optimizer loaded")

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_update_fn
    )
    print(f"{datetime.now()}: scheduler loaded")

    # B,C,d_1,..,d_n expected.
    loss_fn = torch.nn.CrossEntropyLoss(
        label_smoothing=config["label_smoothing"], ignore_index=config["pad_token"]
    )
    print(f"{datetime.now()}: loss_fn loaded")

    metrics = []
    non_smoothed_loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=config["pad_token"]
    )
    non_smoothed_loss_fn.name = "non_smoothed_loss_fn"
    metrics.append(non_smoothed_loss_fn)
    print(f"{datetime.now()}: metrics loaded")

    history = []
    last_epoch_vloss = float("inf")
    non_improve_cnt = 0
    best_val_loss = float("inf")
    start_epoch = -1
    if LOAD_PRE_SAVED:
        print(f"{datetime.now()}: loading pre saved")
        assert path.exists(history_path)
        assert path.exists(optimizer_path)
        with open(history_path, "r") as f:
            dic = json.load(f)
            history = dic["history"]
            last_epoch_vloss = dic["last_epoch_vloss"]
            non_improve_cnt = dic["non_improve_cnt"]
            best_val_loss = dic["best_val_loss"]
            start_epoch = dic["epoch"]
            model_path = dic["model_path"]
        assert path.exists(model_path)


        optimizer.load_state_dict(torch.load(optimizer_path))
        scheduler.load_state_dict(torch.load(scheduler_path))
        model.load_state_dict(torch.load(model_path))
        model.to(
            config["device"], dtype=config["dtype"]
        )

        # fixing the history to be until the last improving epoch
        new_start_epoch = int(re.search(r"model_(\d*)_.*\.pt", model_path).group(1))
        if new_start_epoch != start_epoch:
            start_epoch = new_start_epoch
            idx = len(history)-1
            while idx > 0 and history[idx]["epoch"] > start_epoch: idx-=1
            # the last one should be main history append. 
            print(history[idx])
            best_val_loss = history[idx]["avg_vloss"]
            last_epoch_vloss = best_val_loss
            non_improve_cnt = 0 # last improved is 0. 
            history=history[:idx+1]
        print(
            f"{datetime.now()}: Starting from pre saved with: start_epoch: {start_epoch}, best_val_loss: {best_val_loss:.4f}, last_epoch_vloss: {last_epoch_vloss:.2f}, non_improve_cnt: {non_improve_cnt:.2f}"
        )
    return model,optimizer,scheduler,loss_fn,metrics,history,last_epoch_vloss,non_improve_cnt,best_val_loss,start_epoch

if __name__ == "__main__":
    print(f"{datetime.now()}: code started: ")
    print(f"{datetime.now()}: DEVICE: {DEVICE}")
    if config["benchmark_mode"]:
        print("WARNING: YOU ARE IN BENCH MARK MODE")
    train_loader, val_loader = load_data()

    model, optimizer, scheduler, loss_fn, metrics, history, last_epoch_vloss, non_improve_cnt, best_val_loss, start_epoch = load_model()

    print(f"{datetime.now()}: training started!")
    scaler = torch.amp.GradScaler(DEVICE, enabled=config["use_amp"] and DEVICE != "mps")
    for epoch in range(start_epoch+1, config["epochs"]):
        if epoch > STOP_AFTER_EPOCH:
            break
        model.train(True)
        avg_loss, avg_metrics,avg_vloss,avg_val_metrics,avg_time,epoch_time = train_one_epoch(epoch)

        metrics_text = ", ".join([f"{metrics[i].name}: {avg_metrics[i]:.3f}" for i in range(len(metrics))])
        val_metrics_text = ", ".join([f"{metrics[i].name}_val: {avg_val_metrics[i]:.3f}" for i in range(len(metrics))])
        print(
            f"{datetime.now()}: [epoch {epoch}]: tokens_trained_so_far: {NUM_TOKENS*(epoch+1):,} avg_loss: {avg_loss:.4f}, {metrics_text}, {val_metrics_text}, avg_vloss: {avg_vloss:.4f}, avg_time (s): {avg_time:.3f}, epoch_time (s): {epoch_time:,.3f}"
        )
        history.append({
            "avg_loss": avg_loss,
            "avg_vloss": avg_vloss, 
            "epoch": epoch,
            "avg_time":avg_time,
            "epoch_time":epoch_time
        } | {
            metrics[i].name:avg_metrics[i] for i in range(len(metrics))
        } | {
            metrics[i].name+"_val":avg_val_metrics[i] for i in range(len(metrics))
        }
        )

        if avg_vloss >= last_epoch_vloss:
            non_improve_cnt += 1
            print(
                f"{datetime.now()}: [epoch {epoch}]: epoch did not improve!"
            )
        else:
            non_improve_cnt = 0

        if non_improve_cnt > config["early_stopping_limit"]:
            print(
                f"{datetime.now()}: [epoch {epoch}]: early stopping triggered!"
            )
            break

        last_epoch_vloss = avg_vloss

        is_best=False
        model_path = f"{save_file_path_root}/model_{epoch}_{avg_vloss}.pt"
        if avg_vloss < best_val_loss:
            best_val_loss = avg_vloss
            is_best=True
        else:
            print(f"{datetime.now()}: [epoch {epoch}]: Still not better than max epoch.")
        
        save_hist(
            history_path,
            model_path,
            optimizer_path,
            scheduler_path,
            save_weights=True,
            is_best=is_best
        )
        print(f"{datetime.now()}: [epoch {epoch}]: history saved")
    print(f"{datetime.now()}: Done training! ")