Best BLEU is not BLEU at best epoch, it's best BLEU in general. 

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

