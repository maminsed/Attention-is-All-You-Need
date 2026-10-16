
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
- in dist 2:
```
english_orig_translation = "When I graduated ten years ago, I really thought I already knew everything about the world."
german_text = "Als ich vor zehn Jahren mein Studium abschloss, dachte ich wirklich, ich wüsste bereits alles über die Welt."
```
- in dist 3: 
```
english_orig_translation = "I grew up in a small village where everyone knew everyone and secrets were impossible."
german_text = "Ich bin in einem kleinen Dorf aufgewachsen, in dem jeder jeden kannte und Geheimnisse unmöglich waren."
```
- in dist 4: 
```
english_orig_translation = "The human brain processes visual information much faster than we previously assumed in science."
german_text = "Das menschliche Gehirn verarbeitet visuelle Informationen viel schneller, als wir bisher in der Wissenschaft angenommen haben."
```
- out of dist 2:
```
english_orig_translation = "Write your name in my book."
german_text = "Schreib deinen Namen in mein Buch."
```
- out of dist 3:
```
english_orig_translation = "Last year I went skydiving with my friends. There were cows in the sky."
german_text = "Letztes Jahr bin ich mit meinen Freunden Fallschirm gesprungen. Da waren Kühe am Himmel."
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

## September 19th (v2): 
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

## September 20th (v1):
- reduced number of layers 6 -> 3
- reduced batch_size 64 -> 32

Results:
- improved performance as in it did not overfit up to 10 epochs. 
- BLEU score however, the same as Sep 9. (bs: 64, N:4)

## September 20th (v2):
- **filtered train data (part 1)**
- **sorted val data**


Results:
- Val BLEU went down a bit. However, it improved on smaller sentences.
- It did not overfit. However, Im not so sure if training it for longer is good, because scaling laws says train for less. 


## September 21st:
- Updated BLEU to be GPU efficient and added it to pipeline. Trained up to 15 epochs. 
- (After the run: updated BLEU to be SacreBLEU)

Results:
- Higher BLEU and lower val, early signs of overfitting but again not that much. 
- Sep 9th Ran for 20 epochs and had higher BLEU. 
- For comparision: BLEU increased from 22.1 to 23.7 from 10-15 epochs, Sep9th had 25 with 20 epochs. 
- Personally it looks better to me, but I'm biased. Also it seems like sometimes pass-k will get the right answer even though pass-1 gives straight up bs data. 


## Report on being poor: 
- I trained a bunch of data on a small 11M token dataset for translation and heres what I got out:
- Regularization is still important, but I noticed that going up to 0.5 dropout is a bit extensive. 
- num_enc,num_dec layer of 4 gave the highest performance so far. BLEU 30
- num_enc,num_dec layer of 3 gave closest performance but it still not as good BLEU 20.8. However, it could be trained for much more without overfitting. (rate of improvement was kinda slow tho) - also might be because of batch size
- num_enc,num_dec layer 8 diverged like crazy as in it didn't even try to learn. 
- basically model size is important, it looks like the bigger the better, until you hit a wall. 
- Deep Post layernorm (`LayerNorm(x + sublayer(x))`) are known to be unstable early in training unless warmup is long or you use Pre-LN. bigger models should have lower learning rate during early training (thats because with larger model sizes, even small changes can cause massive shifts and at the start the model is at it's most sensetive point.)
- goign forward only change one param at a time (grid search..?)


## September 27th:
- Getting wmt14 data. -> 10^8
- allowed for different tokenizers. 
- implemented tokenization caching
- Sort Val
- Setup mid epoch save. 
- validation in epoch. 
- Trained until 86800

Results:
- Lowkey didn't perform as good as iwslt on both it's own val set and iwslt's val set. 
    - That might be because IWSLT is out of dist and it's own dataset is way noiser/has more examples than IWSLT so not so much of a fair comparrision. 
    - It has BLEU 18 on it's own and 19 on IWSLT, extremly lower than the 30 baseline. 
- comparing with normal sentence comparision it doesn't seem to be doing as good either. 
- It is important to note that the val loss seems to be increasing and one epoch still hasn't finished. 
- It takes extremly long tho... 

## Oct 6th:
- Tranined up until 1.6 epochs. It genuenlly did not improve that much. 
- BLEU 18 -> 20. which could very much be way smaller change due to BLEU overfitting. But the issue is that non smoothed loss was increasing still.


Next step:
- read the article on how to speed up and come up with ways. 


## To be done:
- optimzie AttnDataset and combine with collate_fn()
- add word dropout
- setup max tokens
- register_buffer for positional encodings. 

- More data.
