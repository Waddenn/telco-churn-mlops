# Production MLOps — IBM Telco Customer Churn

End-to-end binary classification project built for the M2 MLOps course proposal. It trains a leakage-safe scikit-learn pipeline, tunes it with stratified cross-validation, tracks runs and artifacts in MLflow, registers the selected model, produces an auditable evaluation, and serves the fitted pipeline through FastAPI.

## What is included

- versioned public IBM CSV in `data/raw.csv` so a reviewer can test offline;
- reproducible EDA and data-quality report;
- `ColumnTransformer` with numeric imputation/scaling and categorical imputation/one-hot encoding;
- logistic-regression grid search using stratified 5-fold ROC-AUC selection;
- MLflow parameters, child tuning runs, metrics, configuration, analysis, plots, predictions, sklearn model, joblib export, and registry alias `staging`;
- holdout ROC, precision–recall, confusion matrix, six metrics, and row-level error analysis;
- CLI batch prediction, FastAPI service, Docker image, tests, linting, and GitHub Actions CI.

## Reproduce locally

Python 3.11 or 3.12 is recommended.

```bash
make init
make all
```

The individual stages are also explicit:

```bash
make analyze
make train
make evaluate
make test
make lint
```

Outputs are written to `reports/`, `data/processed/`, and `artifacts/`. The fitted pipeline contains all preprocessing, so the same transformations are used in training and inference.

The executed results are summarized in [`reports/model_report.md`](reports/model_report.md); the trained model and evaluation figures are versioned so the API and Docker image can be tested immediately after cloning.

## Inspect experiment tracking

```bash
make mlflow-ui
```

Open <http://127.0.0.1:5000>. The local SQLite backend contains the experiment and model registry. Environment variables in `.env.example` can point the same code at a remote MLflow server.

## Predict

Batch CLI:

```bash
.venv/bin/python -m src.predict --config configs/config.yaml --input example_customer.json
```

API:

```bash
make serve
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data @example_customer.json
```

Interactive OpenAPI documentation is at <http://127.0.0.1:8000/docs>.

## Docker

Train first so `artifacts/model.joblib` exists, then:

```bash
make docker-build
docker run --rm -p 8000:8000 telco-churn-api:latest
```

## Repository structure

```text
mlops-project/
├── .github/workflows/ci.yml
├── artifacts/                 # generated model, metrics, plots, predictions
├── configs/config.yaml
├── data/raw.csv               # IBM dataset (versioned)
├── reports/                   # generated EDA report and figures
├── src/
│   ├── analyze.py
│   ├── api.py
│   ├── evaluate.py
│   ├── pipeline.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── tests/
├── Dockerfile
├── Makefile
└── requirements.txt
```

## Reproducibility and design notes

- The dataset hash and split seed are logged.
- The customer ID is retained for provenance but excluded from features.
- The holdout set is never used by `GridSearchCV`.
- Missing `TotalCharges` values are parsed but imputed only inside training folds.
- `random_state=42` is applied to the split, CV, and estimator.
- Configuration paths are resolved relative to the project, not the shell's current directory.

The source dataset is IBM's public Telco Customer Churn sample: <https://github.com/IBM/telco-customer-churn-on-icp4d/tree/master/data>.
