
- out of dist translation: 
```
english_orig_translation = "I don't know what to say about cats and dogs."
german_text = "Ich weiß nicht, was ich über Katzen und Hunde sagen soll."
```
- in dist translation: 
```
english_orig_translation = "Let's talk about the environment."
german_text = "Lasst uns über die Umwelt sprechen."
```
- out of dist 2:
```
english_orig_translation = "Write your name in my book."
german_text = "Schreib deinen Namen in mein Buch."
```

## results:

## September 9th: 
- Lowered depth from 4 -> 6:
- Increased mega_batch_size from 50 -> 100: pad_tokens/step lowered from 

Results:
- The translation was horrible, even thought it had almost the same val_loss (comparing with Sep 4). They were both 2.6 tho, so Im not even sure if thats helpful.
- However, at 15, the early stopping still hadn't kicked in (not sure if its a good thing or not...)
- out dist: english_pred_translation = "I Finger Finger Finger OverCats and dog systems, I don't know what I need". with BLEU=0.188
- in dist : english_pred_translation = "Let's talk about the environment." with BLEU=1.0
    - This was the first try however, when I kept re-translating it would translate random stuff. A couple examples:
        - "So let's talk about the environment."
        - "So let measureively what eherfronts the environment."
- out dist 2: "So, color your name in my book." BLEU=0.64. It was tryign to say book or something tho. 
- For some reason the model performance improves a lot when theres a period at the end. I think that comes from input data that ends with . 

## Septermber 15th:
- Tried to train up to 20 epochs, however, it didn't improve more than 15. Even the non-smoothed loss seems to be overfitting. 
- Learned about scaling laws of transformers and basically most research talks about number of parameters and data to be on the same scale. They also talk about training with low epochs, because transformers are subject to overfitting. 



- think of a way to perform BLEU efficiently, so you can process it in background, while the next batch is doing it's thing.
    - (check what metric did they use for early stopping)
- try with reduced batch size

## September 19th: 
- BLEU
- reduce epochs to 20->6
- reduce warmup steps 3000 -> 1000
- increase model size if it confirms with scaling laws 
    - number of layers 4 -> 6
- increase dropout. 
    - drop_pre_norm, drop_post_emb 0.1 -> 0.3

dataset size: 10^7, params: 10^7

Results:
- worked horribly. 
- avg_vloss: 6!
- stopped improving at epoch 2...



## To be done:
- update dropout with new dropouts. 
- add BLEU to train.py
- register_buffer for positional encodings. 

- test with lower vocab size
- More data.