PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e .[dev]

update:
	$(BIN)/oil-update-market
	$(BIN)/oil-update-news

features:
	$(BIN)/oil-build-features

train:
	$(BIN)/oil-train-model

signals:
	$(BIN)/oil-generate-signals

report:
	$(BIN)/oil-generate-report

dashboard:
	$(BIN)/streamlit run src/oil_risk/dashboard.py

test:
	$(BIN)/pytest
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .
