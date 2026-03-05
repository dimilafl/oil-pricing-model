# Oil risk premium model

This repository builds a daily Iran oil move risk premium workflow using only free data sources by default. It pulls market and news data, builds features, trains a 3-state regime model, generates daily alerts, runs an evaluation loop, and serves a Streamlit dashboard.

## Quickstart (macOS or Linux)

```bash
git clone https://github.com/dimilafl/oil-pricing-model.git
cd oil-pricing-model

make setup
make daily
make dashboard
```

Local quality gate:

```bash
make smoke
```

## What `make daily` runs

`make daily` executes this sequence:

1. `make update` (market + news)
2. `make features`
3. `make train`
4. `make signals`
5. `make eval`
6. `make report`
7. `make export-alerts`

Equivalent console scripts (installed into the venv):

* `oil-update-market`
* `oil-update-news`
* `oil-build-features`
* `oil-train-model`
* `oil-generate-signals`
* `oil-evaluate-signals`
* `oil-generate-report`
* `oil-export-alerts`

## Commands

* `make setup`
  Create `.venv/` and install `oil-risk` (editable) plus dev tools.

* `make update`
  Run `oil-update-market` and `oil-update-news`.

* `make features`
  Run `oil-build-features`.

* `make train`
  Run `oil-train-model`.

* `make signals`
  Run `oil-generate-signals`.

* `make eval`
  Run `oil-evaluate-signals`.

* `make report`
  Run `oil-generate-report`.

* `make export-alerts`
  Run `oil-export-alerts` and write `artifacts/alerts.json`.

* `make dashboard`
  Run `streamlit run src/oil_risk/dashboard.py`.

* `make test`
  Run pytest + ruff checks.

* `make smoke`
  Fast local gate: ruff + pytest + explicit no-network smoke test.

## Environment variables

* `LOOKBACK_DAYS`
  Lookback window for GDELT DOC API pulls. Default: `90`.

* `POLYGON_API_KEY` (optional)
  Enables options snapshot pull for `USO`, `XLE`, `SPY`.

* `OPENAI_API_KEY` (optional)
  Enables cached one-time news classification stored in `news_llm`.

## Data sources

### Market (FRED, no key)

Pulled and cached to `data/cache/fred_<SERIES>.parquet`.

* WTI: `DCOILWTICO`
* Brent: `DCOILBRENTEU`
* VIX: `VIXCLS`
* OVX: `OVXCLS`
* USD index: `DTWEXBGS`
* 10Y: `DGS10`
* S&P 500 index: `SP500`

### News (GDELT DOC API, no key)

Pulled via the GDELT DOC API. Default lookback is controlled by `LOOKBACK_DAYS` (default 90). Cached to:

* `data/cache/gdelt_api_<days>d_<max_records>r.json`

### Optional modules

* Polygon options snapshot (requires `POLYGON_API_KEY`)
* OpenAI news classification (requires `OPENAI_API_KEY`)

## Storage and outputs

SQLite DB:

* `data/oil_risk.db`

Tables:

* `market_raw(series_id, date, value, source, pulled_at)`
* `market_features(date, feature_name, feature_value)`
* `news_raw(id PK, datetime, source, url, title, raw_record_json, pulled_at)`
* `news_llm(id PK, relevance_score, category, intensity, entities_json, summary, model_name, created_at)`
* `news_features(date, feature_name, feature_value)`
* `options_raw(ticker, date, metric_name, metric_value, source, pulled_at)`
* `options_features(date, feature_name, feature_value)`
* `model_state(date, state_id, state_label, state_probabilities_json)`
* `signals(date, signal_name, signal_value, metadata_json)`
* `signal_eval(date, eval_name, eval_json, created_at)`
* `reports(date, path, created_at)`

Artifacts:

* Daily report: `reports/report_<YYYY-MM-DD>.md`
* Evaluation report: `reports/eval_<YYYY-MM-DD>.md`
* Model artifacts: `models/regime_gmm_<timestamp>.joblib`
* Alerts export: `artifacts/alerts.json`

## What the pipeline computes

### Market features

* `oil_return`, `brent_return` (log diffs)
* `dVIX`, `dOVX`
* `VIX_z_63`, `OVX_z_63`
* `oil_vix_corr_63_proxy`
* If `SP500` present:

  * `spx_return`
  * `oil_spx_corr_63`
* `oil_realized_vol_10d`, `oil_realized_vol_21d`
* `usd_level`, `usd_change`
* `rate_level`, `rate_change`

### News features

Daily aggregates from normalized news output:

* `article_count`, `keyword_count`, `tone_mean`, `negative_tone_magnitude`
* robust z-scores for count and tone components
* `geopolitical_risk_score` = z(article_count) + z(keyword_count) + z(negative_tone_magnitude)
* If LLM rows exist, adds:

  * `intensity_sum`
  * `category_count_<category>`
  * bumps `geopolitical_risk_score` by `0.5 * robust_z(intensity_sum)`

### Options features (only if options_raw exists)

* `put_volume_total`, `call_volume_total`, `put_call_ratio_mean`
* `unusual_put_activity` (max per day, based on robust z of per-ticker put_call_ratio)

### Regime model

* 3-component Gaussian Mixture Model trained on:

  * `oil_return`, `dVIX`, `dOVX`, `usd_change`, `rate_change`, `news_risk_score`
* States labeled `low_risk`, `medium_risk`, `high_risk` by ordering each state's mean `news_risk_score`.
* Writes per-date state and probabilities to `model_state`.

### Signals (latest date only)

Computed from the latest fully-populated feature row:

* `risk_premium_alert`: `OVX_z_63 > 1.0` AND `news_risk_score > 1.0`
* `macro_stress_alert`: `VIX_z_63 > 1.0` AND `oil_return < -0.02`
* `correlation_break_alert`: uses `oil_spx_corr_63` if available else `oil_vix_corr_63_proxy`, triggers when value `< -0.3`
* `hedging_pressure_alert` (only if options features exist): `unusual_put_activity` AND `VIX_z_63 > 1.0`

Note: `signals` is overwritten each run, so it only contains the latest day’s rows.

### Evaluation loop

`oil-evaluate-signals` computes forward return sums from `oil_return`:

* `fwd_1d`, `fwd_5d`, `fwd_10d`

Summaries stored:

* triggered signal summary
* state summary
* state plus signal summary (triggered only)

Note: because `signals` is latest-date only, evaluation output will be thin unless signal history is persisted.

## Dashboard

Run:

```bash
make dashboard
```

Then open:

* [http://localhost:8501](http://localhost:8501)

Dashboard shows:

* WTI, VIX, OVX, SP500 (if present)
* oil SPX correlation (if present)
* news risk and intensity (if present)
* options put/call ratio mean (if present)
* regime scatter timeline
* triggered signals table for the latest date
* renders the latest report and latest evaluation markdown if present

## Windows (PowerShell, no make)

The Makefile assumes a POSIX venv layout (`.venv/bin`). On Windows, run the console scripts after activating the venv.

```powershell
git clone https://github.com/dimilafl/oil-pricing-model.git
cd oil-pricing-model

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"

oil-update-market
oil-update-news
oil-build-features
oil-train-model
oil-generate-signals
oil-evaluate-signals
oil-generate-report
oil-export-alerts

streamlit run src\oil_risk\dashboard.py
```

## Known sharp edges

* FRED and GDELT caches have no TTL. Repeated runs can reuse stale cached payloads unless cache files are removed.
* `signals` is latest-day only, so evaluation depth is limited unless signal history is persisted over time.
