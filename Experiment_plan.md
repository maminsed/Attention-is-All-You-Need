
Test sentences:



### Experiment 1: Initial config
- location: runs/exp_1/
- best epoch: 15
- best BLEU: 0.314
- best non_smoothed_val_loss: 1.65

Personal insights:
- It seemed to be doing pretty well in short texts while performance detriorates as sentences increase in size.
- definetly overfitting after best epoch. 
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
- best epoch: 25
- best BLEU: 0.220
- best non_smoothed_val_loss: 2.08


Personal Insights:
- It seems like that even though it's stable, it seems like that small warmup step was required for the network to achieve better final results, so epoch 1 result in this case was worst. 
- Only tested with greedy decoding and the overall results seemed to be trying to say the same thing however the whole sentence itself does not mean the same thing, you should know what the original sentence was to understand what it's trying to say. 
- Even with this low BLEU short sentences are having correct translations and it messes up with longer sentences. 
