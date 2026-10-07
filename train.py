import json
import math
import random
from datetime import datetime
import time
from os import path
import os
import pickle

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sentencepiece as spm
import torch
from frozendict import frozendict
from torch.nn import ModuleList
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.text import BLEUScore,SacreBLEUScore
import pyarrow as pa

# from torchinfo import summary
from torch.utils.data import DataLoader, Dataset

save_file_path_root = "./test/overnight_Oct_2"
history_path = f"{save_file_path_root}/history.json"
optimizer_path = f"{save_file_path_root}/optimizer.pt"
scheduler_path = f"{save_file_path_root}/scheduler.pt"

LOAD_PRE_SAVED = True
VERBOSE = True
VERBOSE_FREQ = 1000
SAVE_FREQ=20_000
STOP_AFTER_EPOCH=30
PAD_TOKEN = 3
CONTEXT_CAP = 128
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DATAS_SOURCE = "wmt14_curated" # wmt14 or iwslt17 or iwslt17_curated or wmt14_curated


NUM_BATCHES = None

if not path.exists(save_file_path_root):
    os.mkdir(save_file_path_root)

if not path.exists(f"{save_file_path_root}/best/"):
    os.mkdir(f"{save_file_path_root}/best/")


"""
We will run some tests on the IWSLT. We will set baseline and in each experiment only change one var. 
At each Experiment, we take the best var for the previous experiment. 
If it looks epoch = 10 is not enough for a model (it hasn't started converging, we will increase epochs)
BLEU change 1 is just a seed change now. Pick winners by non_smoothed_val_loss 

Baseline:
- vocab_size: 7000
- dataset: iwslt_curated
- filtering_strategy: drop
- epoch: 10
- use_amp: True
- POST_LN

Initial values: 
- num_enc_layers/num_dec_layers: 4
- batch_size: 32, 
- warmup_steps: 3000
- heads: 8,
- drop_rate: 0.1


Experiment 1: 
- everything initial

Experiment 2: 
- PRE_LN

Experiment 3:
phase 1:
- Peak LR -> 0.5x
phase 2:
- Peak LR -> 2x

Experiment 4: N:
phase 1:
- num_enc_layers/num_dec_layers: 3
phase 2:
- N: 2
phase 3: 
- N: 6

Experiment 5: BS:
We will make the peak_LR constant
phase 1:
- batch_size: 64
phase 2: 
- batch_size: 128 (if mac can handle)

Experiment 6: WS:
phase 1:
- WS = 4000
phase 2:
- WS = 1000

"""

config = {
    # model params
    "vocab_size": 37000,
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
    "batch_size": 32,
    "val_batch_size": 32,
    "epochs": 5,
    "early_stopping_limit": 5,
    ## Optimizer
    "optimizer": "adam",  # informational only,
    "lr": 1.0, # scheduler will take care
    "optim_betas": [0.9, 0.98],
    "optim_eps": 1e-9,
    ### Scheduler
    "warmup_steps": 3000, # lowering a bit since were training for less epochs
    ## Loss
    "loss": "CrossEntropyLoss",  # informational only
    "label_smoothing": 0.1,

    # Dataset params
    "context_size": CONTEXT_CAP,
    "pad_token": PAD_TOKEN,
    "test_size": 0.2,
    "val_size": 0.2,
    "mega_batch_size": 100, 
    "data_source": DATAS_SOURCE,
    "filtering_strategy": "drop", # drop or cut. 

    # tokenization
    "tokenizer_source": "wmt14",
    "tokenizer_num_threads": 2,
    "tokenization_cache": True,

    # Device params
    "device": DEVICE,
    "dtype": torch.float32,
    "seed": 42,
    ## Multiproccessing
    "use_amp": True,
    "autocast_dtype": torch.bfloat16 if DEVICE=="mps" else torch.float16,
    "num_workers": 2, # will time. 
    "pin_memory": False,  # doesn't help much in mac and non_blocking weird.
}
assert config["h"] * config["d_v"] == config["d_model"]
assert DEVICE == "mps"
assert config["filtering_strategy"] in ["drop","cut"]
if not config["tokenization_cache"]:
    print("WARNING: TOKENIZATION CACHE IS DISABELED!!! ")

sp_bpe = None
batch_sampler = None

random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])

# generally useful param:
SQRT_INV_DM = 1/math.sqrt(config["d_model"])

# make all temp changes here to not mess up anything else and forget:
if config["benchmark_mode"]:
    config["warmup_steps"] = 12

    config["num_workers"] = 4
    config["use_amp"] = False
    VERBOSE_FREQ=5
    VERBOSE=True


class AttnDataset(Dataset):
    def __init__(self, X: np.array, Y: np.array, X_offset: np.array, Y_offset: np.array):
        super().__init__()
        # Keeping it in cpu so it doesn't face the issues with CUDA or mps and multiprocessing .
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
        self.X_offset = torch.from_numpy(X_offset).long()
        self.Y_offset = torch.from_numpy(Y_offset).long()

    def __len__(self):
        return len(self.X_offset)-1

    def __getitem__(self, idx):
        return (
            self.X[self.X_offset[idx]:self.X_offset[idx+1]], 
            self.Y[self.Y_offset[idx]:self.Y_offset[idx+1]]
        )
    
    def __getitems__(self, indecies):
        return (
            [self.X[self.X_offset[idx]:self.X_offset[idx+1]] for idx in indecies],
            [self.Y[self.Y_offset[idx]:self.Y_offset[idx+1]] for idx in indecies]
        )


class ShuffledNearSampler:
    def __init__(self, data, batch_size,lengths):
        super().__init__()
        self.data:AttnDataset = data
        self.batch_size = batch_size
        self.lengths = lengths

        self.mbi=0
        self.epoch=0

    def set_epoch(self,epoch):
        self.epoch = epoch

    def __iter__(self):
        cur_rng = np.random.default_rng(seed=config["seed"]+self.epoch)
        size = len(self.data)
        meg_size = self.batch_size * config["mega_batch_size"]

        indecies = cur_rng.permutation(size)
        megabatches = [sorted(indecies[i:i+meg_size],key=lambda x :self.lengths[x]) for i in range(0,size,meg_size)]

        megabatches = [[idx[i:i+self.batch_size] for i in range(0,len(idx),self.batch_size)] for idx in megabatches]
        megabatches = [i for megabatch in megabatches for i in megabatch]

        mb_indecies = cur_rng.permutation(len(megabatches))
        for mbi in range(self.mbi,len(mb_indecies)):
            yield megabatches[mb_indecies[mbi]]
        self.mbi=0

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size) - self.mbi

    def load_bit(self,mbi):
        self.mbi = mbi


def collate_fn(batch):
    src, tgt = batch

    src = pad_sequence(src, batch_first=True, padding_value=PAD_TOKEN)
    tgt = pad_sequence(tgt, batch_first=True, padding_value=PAD_TOKEN)

    return src, tgt

def encode_ds(sp_bpe,data_path:str,cache_path:str,nrows=None,num_threads=None):
    if not path.exists(cache_path) or not config["tokenization_cache"]:
        print(f"{datetime.now()} cache does not exist")
        df = pd.read_csv(data_path,nrows=nrows)
        cut=config["filtering_strategy"] == "cut"

        print(f"{datetime.now()} encoding X started: ")
        x_enc = sp_bpe.encode_as_ids(df["detext"].tolist(), add_bos=True, add_eos=not cut, num_threads=num_threads)

        print(f"{datetime.now()} encoding Y started: ")
        y_enc = sp_bpe.encode_as_ids(df["entext"].tolist(), add_bos=True, add_eos=not cut, num_threads=num_threads)

        print(f"{datetime.now()} converting to pyarrow")
        X = pa.array(x_enc,type=pa.large_list(pa.int32())) 
        Y = pa.array(y_enc,type=pa.large_list(pa.int32()))

        num_dropped=0
        if cut:
            raise ValueError("cut is not implemented yet")
        else:
            x_keep = np.diff(X.offsets)  <= CONTEXT_CAP
            y_keep = np.diff(Y.offsets)  <= CONTEXT_CAP
            keep = x_keep & y_keep
            X = X.filter(keep)
            Y = Y.filter(keep)
            num_dropped = int((~keep).sum())
        if config["tokenization_cache"]:
            print(f"{datetime.now()} storing to patable")
            patable = pa.table({
                "Y": Y,
                "X": X,
            }, metadata={
                "CONTEXT_CAP": str(CONTEXT_CAP),
                "filtering_strategy": config["filtering_strategy"],
                "nrows": str(nrows),
                "num_dropped": str(num_dropped)
            })
            pq.write_table(patable,cache_path)
    else:
        print(f"{datetime.now()} cache found")
        patable = pq.read_table(cache_path)
        metadata = {k.decode("utf-8"):v.decode("utf-8") for k,v in patable.schema.metadata.items()}
        if (
            str(CONTEXT_CAP) != metadata["CONTEXT_CAP"] or
            str(config["filtering_strategy"]) != metadata["filtering_strategy"] or
            str(nrows) != metadata["nrows"]
        ):
            raise ValueError("CONTEXT_CAP or filtering_strategy do not match expected")
        num_dropped=0
        X = patable["X"].combine_chunks()
        Y = patable["Y"].combine_chunks()

    return (
        X.values.to_numpy(zero_copy_only=True),
        Y.values.to_numpy(zero_copy_only=True), 
        X.offsets.to_numpy(zero_copy_only=True),
        Y.offsets.to_numpy(zero_copy_only=True), 
        num_dropped
    )


def load_data(sp_bpe) -> tuple[DataLoader, DataLoader]:

    print(f"{datetime.now()}: encoding train started")
    X_train,Y_train,X_train_offset,Y_train_offset,dropped_train = encode_ds(
        sp_bpe,
        f"data/{config['data_source']}/train_df.csv",
        f"data/{config['data_source']}/train_{config["tokenizer_source"]}.parquet",
        nrows=1000 if config["benchmark_mode"] else None,
        num_threads=config["tokenizer_num_threads"],
    )

    print(f"{datetime.now()}: encoding val started")
    X_val, Y_val,X_val_offset,Y_val_offset, dropped_val = encode_ds(
        sp_bpe,
        f"data/{config['data_source']}/val_df.csv",
        f"data/{config['data_source']}/val_{config["tokenizer_source"]}.parquet",
        num_threads=config["tokenizer_num_threads"],
    )

    print(f"{datetime.now()}: # train size: #Tokens: X: {len(X_train):,} Y: {len(Y_train):,} #Sentences: {len(X_train_offset)-1:,}, num_dropped: {dropped_train:,}")
    print(f"{datetime.now()}: # val size: #Tokens: X: {len(X_val):,} Y: {len(Y_val):,} #Sentences: {len(X_val_offset)-1:,}, num_dropped: {dropped_val:,}")

    global NUM_BATCHES
    NUM_BATCHES = math.ceil((len(X_train_offset)-1) / config["batch_size"])
    print(f"{datetime.now()}: batches/epoch: {NUM_BATCHES:,}. LR peak: {SQRT_INV_DM*config["warmup_steps"]**-0.5}")
    if NUM_BATCHES * config["epochs"] * 0.5 < config["warmup_steps"]:
        print(f"{datetime.now()}: WARNING: more than half of epochs are warmup steps. ")

    # setting up the data

    train_set = AttnDataset(X_train, Y_train, X_train_offset, Y_train_offset)
    val_set = AttnDataset(X_val, Y_val, X_val_offset, Y_val_offset)
    # test_set = AttnDataset(X_test,y_test)

    train_lengths=np.diff(Y_train_offset)
    global batch_sampler
    batch_sampler = ShuffledNearSampler(train_set,config["batch_size"],lengths=train_lengths)
    train_loader = DataLoader(
        train_set,
        # batch_size=config["batch_size"],
        # shuffle=True,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        pin_memory=config["pin_memory"],
        num_workers=config["num_workers"],
        persistent_workers=bool(config["num_workers"]),
    )

    val_loader = DataLoader(
        val_set, batch_size=config["val_batch_size"], collate_fn=collate_fn, pin_memory=config["pin_memory"]
    )
    return train_loader, val_loader, len(X_train)


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
        self.embedding = torch.nn.Embedding(self.config["vocab_size"], self.config["d_model"])
        torch.nn.init.uniform_(self.embedding.weight,-SQRT_INV_DM,SQRT_INV_DM)
        self.dropout = torch.nn.Dropout(self.config["drop_post_emb"])

        # encoder
        self.encoder_attention = ModuleList(
            [SelfAttention(self.config) for _ in range(self.config["num_enc_layers"])]
        )
        self.encoder_MLP = ModuleList(
            [MLP(self.config) for _ in range(self.config["num_enc_layers"])]
        )

        # decoder
        self.decoder_attention = ModuleList(
            [
                SelfAttention(self.config, lookahead_mask=True, pad_mask=False)
                for _ in range(self.config["num_dec_layers"])
            ]
        )
        self.cross_attention = ModuleList(
            [SelfAttention(self.config) for _ in range(self.config["num_dec_layers"])]
        )
        self.decoder_MLP = ModuleList(
            [MLP(self.config) for _ in range(self.config["num_dec_layers"])]
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
        # the device names are written next to it assuming that self.config["device"] is CUDA and the rest is the shape. we assume input shape is:
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


def calculate_bleu(loader:DataLoader,model:Transformer,sp_bpe, return_translations:bool=False):
    start_time = time.time()
    print(f"{datetime.now()} started calculating bleu")
    bleu_fn = SacreBLEUScore(lowercase=True)
    model.eval()
    avg_eos_idx = 0
    total_sample = 0
    EOS_ID = sp_bpe.eos_id()
    BOS_ID = sp_bpe.bos_id()
    translations = []
    with torch.no_grad():
        for i,(x, y) in enumerate(loader):
            x = x.to(device=DEVICE, non_blocking=config["pin_memory"])
            B = x.shape[0]
            final_trans = torch.full((B,CONTEXT_CAP), EOS_ID, dtype=torch.long,device=DEVICE) # (B,CP)
            eos_idx = torch.full((B,1), -1,dtype=torch.long,device=DEVICE)

            cur_x = x
            cur_eng_trans = torch.full((B,1),BOS_ID,dtype=torch.long,device=DEVICE) # (B,Sy)
            cur_indecies = torch.arange(B,device=DEVICE)
            num_without_eos=B

            cnt=0
            while cnt < CONTEXT_CAP-1 and cur_x.shape[0] > 0:
                cnt+=1
                output = model(cur_x,cur_eng_trans)
                output = torch.argmax(output[:,-1,:],dim=1,keepdim=True) # B,1

                cur_eng_trans = torch.cat([cur_eng_trans,output],dim=1)

                is_eos = (output.squeeze(1) == EOS_ID)
                if is_eos.any():
                    finished_indecies = cur_indecies[is_eos]
                    eos_idx[finished_indecies] = cnt
                    final_trans[finished_indecies,:cnt+1] = cur_eng_trans[is_eos]
                    num_without_eos-=is_eos.sum()

                    keep = ~is_eos
                    cur_x = cur_x[keep]
                    cur_eng_trans = cur_eng_trans[keep]
                    cur_indecies = cur_indecies[keep]

            if cur_x.shape[0] > 0:
                final_trans[cur_indecies, :cnt+1] = cur_eng_trans
                eos_idx[cur_indecies] = cnt
            
            eos_idx_cpu = eos_idx.squeeze().detach().cpu().tolist()
            avg_eos_idx+=sum(eos_idx_cpu)
            total_sample+=B

            orig_decod = [[sp_bpe.decode_ids(idx)] for idx in y.tolist()]

            final_trans_cpu = final_trans.detach().cpu().tolist()
            eng_decod = []
            for idx in range(B):
                valid_counts = eos_idx_cpu[idx]+1
                eng_decod.append(sp_bpe.decode_ids(final_trans_cpu[idx][:valid_counts]))

            val = bleu_fn(eng_decod,orig_decod)
            if return_translations:
                translations+=[{"orig":orig_decod[idx][0], "trans": eng_decod[idx]} for idx in range(B)]
    end_time = time.time()
    avg_bleu = bleu_fn.compute()
    print(f"{datetime.now()} finished calculating bleu - avg bleu: {avg_bleu*100:.2f} - total_time: {end_time - start_time} - avg_eos_idx: {avg_eos_idx/total_sample:.2f}")
    if return_translations:
        return avg_bleu,total_sample, translations, bleu_fn
    else:
        return avg_bleu

class noop():
    # noops in training. 
    def __init__(self):
        self.name = "bleu_calculations"
    def __call__(self,logits,preds):
        return torch.zeros((),device=DEVICE)

def calculate_val_loss(bleu_calc:bool):
    # val loss:
    model.eval()
    avg_vloss = torch.zeros((),device=DEVICE)
    avg_metrics = torch.zeros((len(metrics),),device=DEVICE)
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device=DEVICE, non_blocking=config["pin_memory"],dtype=torch.long)
            y = y.to(device=DEVICE, non_blocking=config["pin_memory"],dtype=torch.long)

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

    if bleu_calc:
        avg_metrics[-1] = calculate_bleu(val_loader,model,sp_bpe)

    return avg_vloss.item(), [m.item() for m in avg_metrics]

def save_hist(history_path,model_path=None,optim_path=None,scheduler_path=None,*,save_weights=True,is_best=True,batches_done=0):
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
                "config": {k:(v if (
                    not type(v).__module__.startswith("torch")
                    ) else str(v)) 
                    for k,v in config.items()},
                "last_epoch_vloss": str(last_epoch_vloss) if last_epoch_vloss == float("inf") else last_epoch_vloss,
                "non_improve_cnt": non_improve_cnt,
                "best_val_loss": str(best_val_loss) if best_val_loss == float("inf") else best_val_loss,
                "epoch": epoch,
                "last_completed_epoch": last_completed_epoch,
                "model_path": model_path,
                "history_path": history_path,
                "optim_path": optim_path,
                "scheduler_path": scheduler_path,
                "in_epoch_batches": batches_done
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
    prev_end = epoch_start
    batch_sampler.set_epoch(epoch)
    start_batch=batch_sampler.mbi
    if start_batch != 0:
        print(f"{datetime.now()} Starting From starting batch: {start_batch}")
    i=0
    try:
        for idx, (x, y) in enumerate(train_loader):
            i=idx+start_batch
            # x: B,S
            # y: B,Sy
            start_time = time.time()
            x = x.to(device=DEVICE, non_blocking=config["pin_memory"],dtype=torch.long)
            y = y.to(device=DEVICE, non_blocking=config["pin_memory"],dtype=torch.long)
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
                        "data_loading_time": start_time - prev_end,
                        "tokens/sec": ts,
                    }
                )
                if VERBOSE:
                    print(
                        f"{datetime.now()}: [epoch {epoch}] batch {i} loss_value: {loss_value:.4f}, batch_time (s): {delta:,.3f}, data_loading_time (ms): {(start_time - prev_end)*1000:.3f}, tokens/sec: {ts:,.3f}"
                    )

            del logits,loss_value
            if (i+1) % SAVE_FREQ == 0:
                model_path=f"{save_file_path_root}/model_{epoch}_in_batch.pt"
                print(f"{datetime.now()} Saving mid-epoch to: {model_path}")
                
                avg_vloss, avg_val_metrics = calculate_val_loss(False)
                history.append({
                    "avg_vloss": avg_vloss, 
                    "avg_loss": total_loss.item() / (i+1),
                    "epoch": epoch,
                    "batch_idx": i,
                    "batch_percentage": i / NUM_BATCHES
                } | {
                    metrics[i].name:avg_metrics[i].item()/(i+1) for i in range(len(metrics))
                } | {
                    metrics[k].name+"_val":avg_val_metrics[k] for k in range(len(metrics))
                })
                model.train(True)
                print(f"{datetime.now()} Val loss calculation done")
                
                save_hist(
                    history_path,
                    model_path,
                    optimizer_path,
                    scheduler_path,
                    save_weights=True,
                    is_best=False,
                    batches_done=i+1
                )

            prev_end = time.time()
    except KeyboardInterrupt as e:
        print(f"{datetime.now()} keyboard interrupt detected after finishing batches: {i}")
        print(e)
        model_path=f"{save_file_path_root}/model_{epoch}_keyboard_intrrupt.pt"
        save_hist(
            history_path,
            model_path,
            optimizer_path,
            scheduler_path,
            save_weights=True,
            is_best=False,
            batches_done=i
        )
        print(f"{datetime.now()} History saved")
        raise

    epoch_end = time.time()
    
    avg_vloss, avg_val_metrics = calculate_val_loss(epoch%2==0)
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

def load_best_model(latest=False):
    global config
    # ONLY USE IN EVALUATION
    assert path.exists(history_path)
    with open(history_path, "r") as f:
        dic = json.load(f)
        history = dic["history"]
        last_epoch_vloss = float(dic["last_epoch_vloss"])
        non_improve_cnt = dic["non_improve_cnt"]
        best_val_loss = float(dic["best_val_loss"])
        start_epoch = dic["epoch"]
        model_path = dic["model_path"]
        config = {k:(v if k not in ["device","autocast_dtype","dtype"] else config[k]) for k,v in dic["config"].items()}

    model = Transformer(config).to(config["device"], dtype=config["dtype"])
    print("\n\n==============")

    num_parameters=sum(p.numel() for p in model.parameters())
    non_embd_parameters=sum(p.numel() for n,p in model.named_parameters() if n != 'embedding.weight')
    print(f"num_parameters: {num_parameters:,} - num_non_embedding_params: {non_embd_parameters:,}")

    # B,C,d_1,..,d_n expected.
    loss_fn = torch.nn.CrossEntropyLoss(
        label_smoothing=config["label_smoothing"], ignore_index=config["pad_token"]
    )
    print(f"{datetime.now()}: loss_fn loaded")

    print(f"{datetime.now()}: loading best pre saved")
    model_best_path = f"{save_file_path_root}/best/model.pt"
    if path.exists(model_best_path) and not latest:
        print("best model path found.")
        model.load_state_dict(torch.load(model_best_path))
    else:
        print(f"best model path not found. Loading from: {model_path}")
        model.load_state_dict(torch.load(model_path))

    model.to(config["device"], dtype=config["dtype"])

    print(
        f"{datetime.now()}: Starting from pre saved with: start_epoch: {start_epoch}, best_val_loss: {best_val_loss:.4f}, last_epoch_vloss: {last_epoch_vloss:.2f}, non_improve_cnt: {non_improve_cnt:.2f}"
    )
    return model,loss_fn,history


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

    # needs a .name and __call__(logits,preds)
    metrics = []
    non_smoothed_loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=config["pad_token"]
    )
    non_smoothed_loss_fn.name = "non_smoothed_loss_fn"
    metrics.append(non_smoothed_loss_fn)

    bleu_metrics = noop()
    metrics.append(bleu_metrics)



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
            last_epoch_vloss = float(dic["last_epoch_vloss"])
            non_improve_cnt = dic["non_improve_cnt"]
            best_val_loss = float(dic["best_val_loss"])
            start_epoch = dic.get("last_completed_epoch",dic["epoch"])
            model_path = dic["model_path"]
            in_epoch_batches = dic.get("in_epoch_batches",0)
        assert path.exists(model_path)


        optimizer.load_state_dict(torch.load(optimizer_path))
        scheduler.load_state_dict(torch.load(scheduler_path))
        model.load_state_dict(torch.load(model_path))
        batch_sampler.load_bit(in_epoch_batches)

        model.to(
            config["device"], dtype=config["dtype"]
        )

        print(
            f"{datetime.now()}: Starting from pre saved with: start_epoch: {start_epoch}, best_val_loss: {best_val_loss:.4f}, last_epoch_vloss: {last_epoch_vloss:.2f}, non_improve_cnt: {non_improve_cnt:.2f}"
        )
    return model,optimizer,scheduler,loss_fn,metrics,history,last_epoch_vloss,non_improve_cnt,best_val_loss,start_epoch

if __name__ == "__main__":
    print(f"{datetime.now()}: code started: ")
    print(f"{datetime.now()}: DEVICE: {DEVICE}")
    if config["benchmark_mode"]:
        print("WARNING: YOU ARE IN BENCH MARK MODE")
        
    sp_bpe = spm.SentencePieceProcessor()
    sp_bpe.load(f"tokenizers/{config["tokenizer_source"]}/bpe.model")
    assert sp_bpe.vocab_size() == config["vocab_size"]
    train_loader, val_loader, train_size = load_data(sp_bpe)

    model, optimizer, scheduler, loss_fn, metrics, history, last_epoch_vloss, non_improve_cnt, best_val_loss, start_epoch = load_model()

    print(f"{datetime.now()}: training started!")
    scaler = torch.amp.GradScaler(DEVICE, enabled=config["use_amp"] and DEVICE != "mps")
    last_completed_epoch=start_epoch
    for epoch in range(start_epoch+1, config["epochs"]):
        if epoch > STOP_AFTER_EPOCH:
            break
        model.train(True)
        avg_loss, avg_metrics,avg_vloss,avg_val_metrics,avg_time,epoch_time = train_one_epoch(epoch)
        last_completed_epoch=epoch
        
        metrics_text = ", ".join([f"{metrics[i].name}: {avg_metrics[i]:.3f}" for i in range(len(metrics))])
        val_metrics_text = ", ".join([f"{metrics[i].name}_val: {avg_val_metrics[i]:.3f}" for i in range(len(metrics))])
        print(
            f"{datetime.now()}: [epoch {epoch}]: tokens_trained_so_far: {train_size*(epoch+1):,} avg_loss: {avg_loss:.4f}, {metrics_text}, {val_metrics_text}, avg_vloss: {avg_vloss:.4f}, avg_time (s): {avg_time:.3f}, epoch_time (s): {epoch_time:,.3f}"
        )
        history.append({
            "avg_loss": avg_loss,
            "avg_vloss": avg_vloss, 
            "epoch": epoch,
            "avg_time":avg_time,
            "epoch_time":epoch_time,
            "cpu_alloc_memory": torch.mps.current_allocated_memory()/torch.mps.driver_allocated_memory(),
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