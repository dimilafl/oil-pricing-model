# Oil risk premium model

This repository builds a daily Iran oil move risk premium workflow using only free data sources.

## Requirements
- Python 3.11+
- Internet access for FRED and GDELT pulls

## Quickstart
```bash
make setup
make update
make features
make train
make signals
make report
```

Run tests and lint checks:
```bash
make test
```

Run dashboard:
```bash
make dashboard
```

## Data sources
- FRED CSV endpoint for DCOILWTICO, DCOILBRENTEU, VIXCLS, OVXCLS, DTWEXBGS, DGS10
- GDELT GKG 2.1 zipped feed from the public last update list

## Storage
- SQLite database at `data/oil_risk.db`
- Raw cache files in `data/cache`
- Reports in `reports`
- Model artifacts in `models`

## Verification commands executed
```bash
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/oil-update-market
.venv/bin/oil-update-news
.venv/bin/oil-build-features
.venv/bin/oil-train-model
.venv/bin/oil-generate-signals
.venv/bin/oil-generate-report
```
