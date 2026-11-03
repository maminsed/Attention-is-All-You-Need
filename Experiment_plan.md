
Test sentences:



### Initial config:
- location: runs/overnight_Nov_3/
- best epoch: 13
- best BLEU: 0.318
- best smoothed_val_loss: 1.65

Personal insights:
- It seemed to be doing pretty well in short texts while performance detriorates as sentences increase in size.
- definetly overfitting after best epoch. 
- performance seems extremly fast on CUDA. 
