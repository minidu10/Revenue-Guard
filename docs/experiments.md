# Experiment Log

## Baseline: DummyClassifier (most frequent)
accuracy: 0.73  recall: 0.00
Catches zero churners. This is the number to beat.

## Experiment 1: Logistic Regression + scaling + one-hot, class_weight=balanced
ROC-AUC:0.8416388953473353
recall:0.783
precision:0.504
accuracy:0.738
Notes:

Threshold	Recall	Precision	Missed leavers(fn)	False alarms (fp)
0.6	        0.706	0.540	    110	                225
0.5	        0.783	0.504	    81	                288
0.4	        0.866	0.466	    50	                371