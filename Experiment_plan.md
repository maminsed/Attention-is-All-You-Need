Best BLEU is not BLEU at best epoch, it's best BLEU in general. 

### General Idea
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
phase 1: 
- PRE_LN
phase 2:
- PRE LN without warmup steps.

Experiment 3:
phase 1:
- Peak LR -> 0.5x
phase 2:
- Peak LR -> 2x

Experiment 4: BS:
phase 1:
- batch_size: 64
phase 2: 
- batch_size: 128 (if compute can handle it)
phase 3:
- batch_size: 128
- same peak_lr but lower warmup steps to be 1/4 of the original wamrup steps so number of updates seen is kinda comparable
phase 4:
- batch_size: 128
- same peak_lr as p2,3
- 1/4 warmup steps
- slow_decay tau: 3000
phase 5:
- batch_size: 32
- same peak_lr
- 3k warmup steps
- slow_decay tau: 5k

Experiment 5: N:
phase 1:
- num_enc_layers/num_dec_layers: 3
phase 3: 
- N: 6

Experiment 6: dropout:
phase 1: 
- dropout: 0.3
phase 2:
- dropout: 0.3
- source_word_dropout: 0.1
phase 3:
- dropout: 0.3
- source_word_dropout: 0.1
- target_word_dropout: 0.1

### Experiment 1: Initial config
- location: runs/exp_1/
- best epoch: 15
- best BLEU: 0.313
- best non_smoothed_val_loss: 1.65

Personal insights:
- It seemed to be doing pretty well in short texts while performance detriorates as sentences increase in size.
- mild overfitting (I guess not too much rise)
- performance seems extremly fast on CUDA. 


### Experiment 2 - Phase 1: PRE-LN
- location: runs/exp_2_phase_1/
- best_epoch: 13
- best BLEU: 0.310
- best non_smoothed_val_loss: 1.65


Personal insights:
- we have lower training loss however, after 13 it seems to be more prone to overfitting. 
- It reached better val values faster and the difference between the BLEU values seem to be too small to be from actual difference. 
- To be honest, I can't tell the difference between this one and exp 1 from the test sentences I run. 
    - They both seem to do decent with multi-shot calls -> this only when not doing greedy decoding. after greedy decoding the sentence does not change
    - They both do worse for longer sentences (understandable)
    - The translations almost have all the correct letters, it just doesn't make sense (sampeled decoding)
    - with greedy decoding it's performing much better, however, it seems to be missing some filler words such as "already", etc. 


### Experiment 2 - phase 2: PRE-lN + no warmup. 
- location: runs/exp_2_phase_2/
- best epoch: 24
- best BLEU: 0.220
- best non_smoothed_val_loss: 2.08


Personal Insights:
- It seems like that even though it's stable, it seems like that small warmup step was required for the network to achieve better final results, so epoch 2 val result in this case was worst. 
- Only tested with greedy decoding and the overall results seemed to be trying to say the same thing however the whole sentence itself does not mean the same thing, you should know what the original sentence was to understand what it's trying to say. 
- Even with this low BLEU short sentences are having correct translations and it messes up with longer sentences. 


### Experiment 3 - phase 1: 0.5x 
- location: runs/exp_3_phase_1/ (has a part 2)
- best epoch: 11
- best BLEU: 0.311
- best non_smoothed_val_loss: 1.66


Personal Insights:
- Kinda same as Exp2-Phase1
- at this point I feel like the change in the BLEU and non_smoothed_val_loss is too small for me to see anything. 
- training loss did decrease faster tho while val loss lagged behind. 
- Same manual evaluation as before. 


### Experiment 3 - phase 2: 2x
- location: runs/exp_3_phase_2/
- best epoch: 24
- best non_smoothed_val_loss: 2.90
- best BLEU: 


Personal Insights:
- even worse validation behaviour than PRE-lN + no warmup. 
- it did not overfit but it takes extremly long to train and batch loss has higher instability

*Feature Implemented* -> broke down forward into encode and decode for faster val/inference. 

### Experiment 4 - phase 1: 64 batch_size
- location: runs/exp_4_phase_1/
- best_epoch: 13
- best non_smoothed_val_loss: 1.64
- best BLEU: 0.313


Perosnal Insight:
- better performance than initial config and exp2_phase1, but with way less epoch total time
- manual testing showed it still has the same problems, which makes me think its also remotly the issue of data. 



### Experiment 4 - phase 2: 128 batch_size
- location: runs/exp_4_phase_2/
- best_epoch: 11
- best non_smoothed_val_loss: 1.632
- best BLEU: 0.315


Personal Insight:
- even better performance than exp4 phase 1 with less total epoch time
- manual testing still shows the same thing tho. 
- it seems to have higher early batch_loss and non smoothed val loss at first which is why Im thinking a higher peak_lr could be helpful. 


### Experiment 4 - phase 3: 128 batch_size, but same warmup (tokens seens) and peak_lr
- location: runs/exp_4_phase_3/
- best_epoch: 9
- non_smoothed_val_loss: 1.686
- best BLEU: 0.305

Personal Insights: 
- while it has better early training behaviour than exp_4_phase_2, it has worse end training behaviour than it and overfits earlier. 
- Moreover, it also performs worse than exp2_p1
- This points to the fact that maybe the improvement in exp4_p2 was because we had higher learning rate later in training
- these all point to exp4_p4 which would studies a learning rate that decays slower


*Feature Implemented* -> kv_cache

### Experiment 4 - phase 4: 128 batch_size, same warmup as phase 3, slower decay
- best_epoch: 11
- best non_smoothed_val_loss: 1.6487
- best BLEU: 0.3113


Personal Insights:
- It's performing better early loss compared to exp4-p2 and better late stage score compared to exp4-p3. 
- However, it still cannnot match the exp4-p2 performance. 
- I would say it's performing similar to exp2-p1's perfomance. 
- I believe this lack of performance is because of too little optimizer steps, which is why for the final phase of this experiment group, I will do bs=32, and a slower decay

### Experiment 4 - phaes 5: 32 batch_size, slower_decay, tau: 5k
- best epoch: 13
- best non_smoothed_val_loss: 1.659
- best BLEU: 0.308

Personal Insights:
- It seems like this is performing worse than p4 and p2 and exp2-p1, Im guessing this doesn't support that in batch size 32 higher lr neccessarily helps. 
- Im assuming maybe thats because batch size 32 has higher noise and thats the reason high lr is not neccessarily a good thing, while bs 128 reduces that noise a lot so it can use higher lr. 
- I guess the whole experiment 4 also showed that early performance and late performance does not always go hand in hand. 


### Experiment 5 - phase 1: (exp4-phase2 cont.) num layers: 3
- best epoch: 11
- best non_smoothed_val_loss: 1.669
- best BLEU: 0.3099

Personal Insights:
- Worse performance than benchmark, kinda similar performance to exp2-p1 (even worse)
- Im guessing scaling laws...


### Experiment 5 - phase 2: num layers: 6
- best epoch: 9
- best non_smoothed_val_loss: 1.604
- best BLEU: 0.3197

Personal Insights: 
- Achieves better performance in lower epochs, but also as a result it's more suseptible to overfitting, so the rate of increase in val loss is much faster when it's overfitting. 
- Best so far. winner of this experiment. 
- Again scaling laws coming in effect, however, this is making the 20x train data harder. 
- 29M params -> 44M params, almost double while tokens stay at 5M tokens, 1/9


### Experiment 6 - phase 1: dropout -> 0.3
- best epoch: 24 (/34)
- best non_smoothed_val_loss: 1.505
- best BLEU: 0.3320

Personal Insight:
- we saw in exp5-p2 that it has very high potential but it overfits early, so I think higher dropout was a really good choice because it allowed the model to not overfit and train for longer, hence a higher epoch. 
- It did have lower train performance as expected and worse early training performnace. which are all expected.
- Im assuming these are a lot would change with the scaling laws so I will be careful with wmt_14


### Experiment 6 - phase 2: source word dropout -> 0.1
- best epoch: 30 (/34)
- best non_smoothed_val_loss: 1.491
- best BLEU: 0.3304

Personal Insights:
- Again higher performance, allowed for less overfitting, however, it kicked in kinda late in training (epoch 21) and up until that point it was worse than exp6-p1. 
- It did have lower BLEU tho which is interesting. Im assuming BLEU wouldn't see improvement unless the top word increased while loss would see improvement if not top decreased which is what dropout might be teaching...?


### Experiment 6 - phase 3: target word dropout -> 0.1
- best epoch: 38 (/40)
- best non_smoothed_val_loss: 1.4673
- best BLEU: 0.3390

Personal Insights:
- best performance so far. However, it didn't pass previous best until epoch 25. 
- I believe the longer sequences have improved a lot, but it's missing a lot of the stuff like everyone, even, etc. (not exactly main parts of sentence but needed for sentence to make sense) - sometimes it uses synonyms but Im greedy decoding. 
- Entire experiments are done so most of what I learned:
    - early loss behaviour does not determine end loss behaviour
    - the larger the network the better performance (at least up to 40M layers) However its extremly more prone to overfitting. 
    - When your data is sparse, dropout is your best friend. 
    - Batch size is a complicated thing, longer batch sizes give better performance when you combine them with higher lr (from what I've seen) because the gradients are more stable. 
    - PRE_LN performs better, but it's not as stable that it does not need a warmup. 
