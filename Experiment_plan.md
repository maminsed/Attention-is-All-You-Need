
Test sentences:



### Experiment 1: Initial config
- location: runs/overnight_Nov_3/
- best epoch: 15
- best BLEU: 0.314
- best non_smoothed_val_loss: 1.65

Personal insights:
- It seemed to be doing pretty well in short texts while performance detriorates as sentences increase in size.
- definetly overfitting after best epoch. 
- performance seems extremly fast on CUDA. 


### Experiment 2 - Phase 1: PRE-LN
- location: runs/overnight_Nov_3_exp_2/
- best_epoch: 13
- best BLEU: 0.310
- best non_smoothed_val_loss: 1.65


Personal insights:
- we have lower training loss however, after 13 it seems to be more prone to overfitting. 
- It reached better val values faster and the difference between the BLEU values seem to be too small to be from actual difference. 
- To be honest, I can't tell the difference between this one and exp 1 from the test sentences I run. 
    - They both seem to do decent with multi-shot calls
    - They both do worse for longer sentences (understandable)
    - The translations almost have all the correct letters, it just doesn't make sense
