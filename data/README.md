# Dataset

`raw.csv` is the public IBM Telco Customer Churn sample dataset (7,043 customers). It is versioned deliberately so the evaluator can run the project locally without credentials or a download step.

- Source: https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv
- SHA-256: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`

The training code converts whitespace-only values in `TotalCharges` to missing numeric values. Imputation then occurs inside each training fold, never before the split.
