# Model evaluation report

## Experiment

- MLflow run ID: `4a34323637614916b882d4078a959f91`
- Model: logistic regression after stratified 5-fold grid search
- Selection metric: ROC-AUC
- Best mean CV ROC-AUC: **0.8463**
- Untuned baseline holdout ROC-AUC: **0.8419**
- Tuned model holdout ROC-AUC: **0.8411**

The holdout set was not used for model or hyperparameter selection. The small difference between baseline and tuned holdout ROC-AUC is reported transparently; the selected parameters are those with the best cross-validation score on the training set.

## Selected hyperparameters

| Parameter | Value |
|---|---:|
| `model__C` | `10.0` |
| `model__class_weight` | `None` |
| `model__solver` | `liblinear` |

## Final holdout metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.8048 |
| Precision | 0.6552 |
| Recall | 0.5588 |
| F1 | 0.6032 |
| Roc Auc | 0.8411 |
| Average Precision | 0.6281 |

## Artifacts and error analysis

- `artifacts/roc_curve.png`
- `artifacts/precision_recall_curve.png`
- `artifacts/confusion_matrix.png`
- `artifacts/predictions.csv` (input rows, actual labels, predictions, probabilities, and error flag)
- `artifacts/model.joblib` (full preprocessing and estimator pipeline)

All metrics and artifacts above are also attached to the same finished MLflow run. The registered model is available as `ChurnClassifier@staging`.
