PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
PYTHON_BOOTSTRAP ?= python3.12
CONFIG ?= configs/config.yaml
EXP ?= telco-churn-production

.PHONY: init analyze train evaluate test lint format all serve mlflow-ui docker-build clean

init:
	$(PYTHON_BOOTSTRAP) -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

analyze:
	$(PYTHON) -m src.analyze --config $(CONFIG)

train:
	MLFLOW_EXPERIMENT_NAME=$(EXP) $(PYTHON) -m src.train --config $(CONFIG)

evaluate:
	MLFLOW_EXPERIMENT_NAME=$(EXP) $(PYTHON) -m src.evaluate --config $(CONFIG)

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m ruff format src tests

all: analyze train evaluate test lint

serve:
	$(PYTHON) -m uvicorn src.api:app --host 0.0.0.0 --port 8000

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

docker-build:
	docker build -t telco-churn-api:latest .

clean:
	rm -rf data/processed/* artifacts/* reports/figures/* mlruns .pytest_cache .ruff_cache
