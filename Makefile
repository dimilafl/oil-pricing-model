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

eval:
	$(BIN)/oil-evaluate-signals

export-alerts:
	$(BIN)/oil-export-alerts

export-signal-engine:
	$(BIN)/oil-export-signal-engine

dashboard:
	$(BIN)/streamlit run src/oil_risk/dashboard.py

test:
	$(BIN)/pytest
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:
	$(BIN)/ruff check . --fix
	$(BIN)/ruff format .

daily:
	$(MAKE) update
	$(MAKE) features
	$(MAKE) train
	$(MAKE) signals
	$(MAKE) eval
	$(MAKE) report
	$(MAKE) export-alerts
	$(MAKE) export-signal-engine

smoke:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .
	$(BIN)/pytest
	$(BIN)/pytest tests/test_no_network_smoke.py
