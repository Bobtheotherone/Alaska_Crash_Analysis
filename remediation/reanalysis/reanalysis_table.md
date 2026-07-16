# Re-analysis of the paper's four confusion matrices (ORIGINAL contaminated protocol)

Test set: N = 10936  (class 0 = 7395, class 1 = 3152, class 2 = 389).

Source: result screenshots in the final report (random 80/20 split, `OneHotEncoder(min_frequency=0.01)`,
no baselines, no uncertainty). `oMAE`/`2-step` lower is better. `pred-0 share` = fraction predicted class 0.
Convention (F1-CONV-001): for a class that is never predicted, precision is *undefined* (shown as **—**),
while recall and F1 are measured zeros under the count-based definition F1 = 2TP/(2TP+FP+FN);
macro-F1 averages the class F1 values including those zeros.

| Model | Acc | oMAE↓ | QWK | macroF1 | balAcc | ≤1acc | 2-step↓ | sevP | sevR | sevF1 | pred-0 share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Majority baseline | 0.6762 | 0.3594 | 0.0000 | 0.2689 | 0.3333 | 0.9644 | 0.0356 | — | 0.0000 | 0.0000 | 1.0000 |
| Decision Tree | 0.5797 | 0.4510 | 0.2210 | 0.4515 | 0.4718 | 0.9693 | 0.0307 | 0.2345 | 0.3393 | 0.2773 | 0.6095 |
| XGBoost | 0.6274 | 0.4013 | 0.3674 | 0.5142 | 0.5781 | 0.9713 | 0.0287 | 0.2518 | 0.5424 | 0.3439 | 0.5738 |
| MLRF (RandomForest) | 0.7098 | 0.3089 | 0.3650 | 0.5064 | 0.5094 | 0.9813 | 0.0187 | 0.4027 | 0.3830 | 0.3926 | 0.8667 |
| EBM | 0.5785 | 0.4902 | 0.3107 | 0.4604 | 0.5829 | 0.9313 | 0.0687 | 0.1585 | 0.6812 | 0.2572 | 0.5336 |

## Key observations
* The trivial majority-class predictor scores **0.6762 accuracy**. Below-majority models: **Decision Tree, XGBoost, EBM** (3 of 4).
* No model is uniformly dominant: MLRF leads accuracy and ordinal MAE (by heavily predicting class 0 — a 86.7% class-0 share); EBM leads severe recall (0.681) but at 0.158 severe precision; XGBoost leads macro-F1.
* A declared primary objective is therefore indispensable; 'best model' is undefined here.
