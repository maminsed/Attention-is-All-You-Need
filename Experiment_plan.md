
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

## September 9th: 
- Lowered depth from 4 -> 6:
- Increased mega_batch_size from 50 -> 100: pad_tokens/step lowered from 

### results:
### Up until epoch 15:
- The translation was horrible, even thought it had almost the same val_loss (comparing with Sep 4). They were both 2.6 tho, so Im not even sure if thats helpful.
- However, at 15, the early stopping still hadn't kicked in (not sure if its a good thing or not...)
- out dist: english_pred_translation = "I Finger Finger Finger OverCats and dog systems, I don't know what I need". with BLEU=0.188
- in dist : english_pred_translation = "Let's talk about the environment." with BLEU=1.0
    - This was the first try however, when I kept re-translating it would translate random stuff. A couple examples:
        - "So let's talk about the environment."
        - "So let measureively what eherfronts the environment."
- out dist 2: "So, color your name in my book." BLEU=0.64. It was tryign to say book or something tho. 
- For some reason the model performance improves a lot when theres a period at the end. I think that comes from input data that ends with . 

## To be done:
- reduce batch_size, needs experimenting

- update dropout with new dropouts. 
- add BLEU to train.py
- register_buffer for positional encodings. 

- test with lower vocab size