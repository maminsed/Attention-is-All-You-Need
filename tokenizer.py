import train
import sentencepiece as spm
import os.path as path
from datetime import datetime
import pandas as pd

if __name__ == "__main__":
    DATA_SOURCE = train.DATAS_SOURCE
    TOKENIZER_SOURCE = train.config["tokenizer_source"]

    TOKENIZER_INPUT = f"data/{DATA_SOURCE}/alltext.txt"

    test_sentence="Hi this is a test sentence"
    if not path.exists(TOKENIZER_INPUT):
        print(f"{datetime.now()} Tokenizer input does not exist, have to create a new one")
        df = pd.read_csv(f"data/{DATA_SOURCE}/train_df.csv")
        with open(TOKENIZER_INPUT,"w") as f:
            for idx,row in df.iterrows():
                f.write(row["detext"]+"\n")
                f.write(row["entext"]+"\n")
        del df

    print(f"{datetime.now()} Training tokenizer started")
    spm.SentencePieceTrainer(
        f"--model_type=bpe --vocab_size={train.config["vocab_size"]} --input={TOKENIZER_INPUT} --mode_prefix=tokenizers/{TOKENIZER_SOURCE}/bpe --pad_id={train.PAD_TOKEN}"
    )
    print(f"{datetime.now()} Training tokenizer finished")

    sp_bpe = spm.SentencePieceProcessor()
    sp_bpe.load(f"tokenizers/{TOKENIZER_SOURCE}/bpe.model")

    print(f"{datetime.now()} Test encoding: {test_sentence}={sp_bpe.encode_as_ids(test_sentence,add_bos=True,add_eos=True)}")
    