# Agentic CSV Data Scientist

An end-to-end CSV analysis app that combines data ingestion, structural audit, cleaning, EDA, outlier and anomaly checks, feature engineering, baseline modeling, explainability, and report export inside a Streamlit interface.

The app is designed to work even when the local LLM is unavailable. Ollama is preferred for planner and narrative generation, but the pipeline now falls back to deterministic summaries so the CSV workflow can still complete.

## What It Does

- Upload a CSV, TSV, or TXT dataset from the web app
- Detect schema roles such as target, numeric, categorical, datetime, text, and ID-like columns
- Run profiling, missingness checks, uniqueness checks, cleaning, EDA, outlier analysis, and anomaly detection
- Train a baseline model when a supervised target is available
- Generate explainability, insight, critique, and recommendation text
- Show the workflow step by step in the UI
- Export a markdown and HTML master report plus datasets, charts, metadata, and model artifacts

## UI Highlights

- Two app themes: `light` and `dark`
- Live workflow timeline with stage status
- Structural audit, quality, profile, modeling, narrative, and download views
- Runtime health display for required packages, optional packages, and Ollama connectivity
- Debug JSON hidden behind optional expanders

## Project Layout

```text
agents/        LLM-backed and fallback planning/narrative agents
analytics/     Ingestion, profiling, cleaning, EDA, splitting, modeling
artifacts/     Export helpers for datasets, charts, metadata, and models
config/        Settings, paths, model config, package flags
core/          Pipeline orchestration, workflow state, contracts, validation
reporting/     Markdown and HTML report builders
sample_data/   Small sample CSV files for quick testing
ui/            Streamlit UI, components, and sidebar controls
tests/         Smoke, unit, and integration coverage
```

## Quick Start

### 1. Create an environment

Python `3.11+` is required.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Optional: configure Ollama

Copy `.env.example` to `.env` and update if needed:

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
APP_ENV=development
```

If Ollama is not running, the app will use fallback planner/narrative logic instead of failing the whole run.

### 4. Start the app

```bash
python run_app.py
```

This launches:

```bash
python -m streamlit run ui/streamlit_app.py
```

## How To Use

1. Open the Streamlit app in the browser.
2. Upload a CSV file.
3. Choose the target column from the sidebar or leave it on `AUTO`.
4. Pick `light` or `dark` theme.
5. Run the full pipeline.
6. Review the workflow timeline, structural audit, model results, and generated narrative.
7. Download reports and artifacts from the `Downloads` tab.

## Outputs

Run outputs are stored under `outputs/` and per-run artifacts are organized under `outputs/runs/<run_id>/`.

Typical outputs include:

- `master_report.md`
- `master_report.html`
- cleaned dataset exports
- train/test exports when modeling runs
- chart images
- metadata and run manifest JSON
- model and preprocessor artifacts when available

## Testing

Run the test suite with:

```bash
python -m pytest -q
```

If the environment is missing core packages such as `pandas` or `streamlit`, install `requirements.txt` first.

## Notes

- Supervised modeling runs only when a valid target is available.
- Explainability runs only when a model is trained.
- Some advanced analytics depend on optional libraries; the app shows their availability in the runtime health panel.
- The exported master report is notebook-inspired, but intentionally lighter than a full notebook parity build.
