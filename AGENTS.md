# Agent instructions

## Scope
These instructions apply to the full repository.

## Workflow rules
- Use Python 3.11 or newer.
- Keep all data pulls reproducible and keyless by default.
- Persist long term data in SQLite at `data/oil_risk.db`.
- Cache raw pulled files in `data/cache/` as Parquet or raw text snapshots.
- Prefer small, modular pipeline scripts under `src/oil_risk/pipelines/`.
- Write or update tests for each adapter and feature logic change.
- Run `make test` before opening a pull request.
- For workflow checks, run `make update`, `make features`, `make train`, `make signals`, and `make report`.

## Commands
- `make setup`
- `make update`
- `make features`
- `make train`
- `make signals`
- `make report`
- `make dashboard`
- `make test`
