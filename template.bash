#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="."

echo "[INFO] Creating project scaffold..."

DIRS=(
  "config"
  "app"
  "ui"
  "core"
  "agents"
  "integrations"

  "analytics"
  "analytics/ingestion"
  "analytics/schema"
  "analytics/profiling"
  "analytics/cleaning"
  "analytics/eda"
  "analytics/outliers"
  "analytics/anomalies"
  "analytics/feature_engineering"
  "analytics/splitting"
  "analytics/modeling"
  "analytics/evaluation"
  "analytics/explainability"

  "reporting"
  "artifacts"
  "utils"
  "prompts"
  "docs"

  "tests"
  "tests/unit"
  "tests/integration"
  "tests/smoke"

  "sample_data"

  "outputs"
  "outputs/reports"
  "outputs/charts"
  "outputs/datasets"
  "outputs/models"
  "outputs/pipelines"
  "outputs/metadata"
  "outputs/runs"
)

FILES=(
  "README.md"
  "requirements.txt"
  "pyproject.toml"
  ".env.example"
  ".gitignore"
  "template.py"
  "run_app.py"
  "run_tests.py"

  "config/__init__.py"
  "config/settings.py"
  "config/paths.py"
  "config/model_config.py"
  "config/package_flags.py"
  "config/report_config.py"

  "app/__init__.py"
  "app/main.py"
  "app/session_manager.py"
  "app/startup.py"

  "ui/__init__.py"
  "ui/streamlit_app.py"
  "ui/sidebar.py"
  "ui/home_page.py"
  "ui/overview_tab.py"
  "ui/data_quality_tab.py"
  "ui/eda_tab.py"
  "ui/outlier_tab.py"
  "ui/feature_engineering_tab.py"
  "ui/modeling_tab.py"
  "ui/explainability_tab.py"
  "ui/insights_tab.py"
  "ui/artifacts_tab.py"
  "ui/components.py"

  "core/__init__.py"
  "core/state.py"
  "core/enums.py"
  "core/contracts.py"
  "core/exceptions.py"
  "core/pipeline.py"
  "core/router.py"
  "core/executor.py"
  "core/retry_manager.py"
  "core/validation_manager.py"
  "core/run_context.py"

  "agents/__init__.py"
  "agents/base_agent.py"
  "agents/planner_agent.py"
  "agents/insight_agent.py"
  "agents/critic_agent.py"
  "agents/recommendation_agent.py"
  "agents/explanation_agent.py"
  "agents/prompt_builder.py"
  "agents/prompt_guard.py"
  "agents/response_parser.py"

  "integrations/__init__.py"
  "integrations/ollama_client.py"
  "integrations/duckdb_client.py"
  "integrations/ydata_profile_adapter.py"
  "integrations/feature_engine_adapter.py"
  "integrations/pyod_adapter.py"
  "integrations/imblearn_adapter.py"
  "integrations/shap_adapter.py"
  "integrations/cleanlab_adapter.py"
  "integrations/evidently_adapter.py"
  "integrations/package_healthcheck.py"

  "analytics/__init__.py"

  "analytics/ingestion/__init__.py"
  "analytics/ingestion/loader.py"
  "analytics/ingestion/encoding_detector.py"
  "analytics/ingestion/delimiter_detector.py"
  "analytics/ingestion/file_validator.py"
  "analytics/ingestion/header_normalizer.py"

  "analytics/schema/__init__.py"
  "analytics/schema/schema_detector.py"
  "analytics/schema/dtype_inference.py"
  "analytics/schema/id_detector.py"
  "analytics/schema/datetime_detector.py"
  "analytics/schema/text_detector.py"
  "analytics/schema/target_inference.py"

  "analytics/profiling/__init__.py"
  "analytics/profiling/profiler.py"
  "analytics/profiling/summary_stats.py"
  "analytics/profiling/missing_profile.py"
  "analytics/profiling/duplicates_profile.py"
  "analytics/profiling/uniqueness_profile.py"
  "analytics/profiling/memory_profile.py"
  "analytics/profiling/profile_formatter.py"

  "analytics/cleaning/__init__.py"
  "analytics/cleaning/cleaner.py"
  "analytics/cleaning/null_normalizer.py"
  "analytics/cleaning/duplicate_handler.py"
  "analytics/cleaning/dtype_fixer.py"
  "analytics/cleaning/constant_column_handler.py"
  "analytics/cleaning/whitespace_cleaner.py"
  "analytics/cleaning/invalid_value_handler.py"
  "analytics/cleaning/cleaning_log.py"

  "analytics/eda/__init__.py"
  "analytics/eda/eda_runner.py"
  "analytics/eda/univariate.py"
  "analytics/eda/bivariate.py"
  "analytics/eda/target_analysis.py"
  "analytics/eda/correlation_analysis.py"
  "analytics/eda/skewness_analysis.py"
  "analytics/eda/multicollinearity.py"
  "analytics/eda/segmentation_hints.py"

  "analytics/outliers/__init__.py"
  "analytics/outliers/outlier_router.py"
  "analytics/outliers/column_outlier_detector.py"
  "analytics/outliers/iqr_detector.py"
  "analytics/outliers/zscore_detector.py"
  "analytics/outliers/mad_detector.py"
  "analytics/outliers/winsorizer.py"
  "analytics/outliers/outlier_registry.py"
  "analytics/outliers/outlier_scorer.py"
  "analytics/outliers/treatment_policy.py"
  "analytics/outliers/outlier_report.py"

  "analytics/anomalies/__init__.py"
  "analytics/anomalies/anomaly_runner.py"
  "analytics/anomalies/isolation_forest_detector.py"
  "analytics/anomalies/lof_detector.py"
  "analytics/anomalies/ecod_detector.py"
  "analytics/anomalies/anomaly_registry.py"
  "analytics/anomalies/anomaly_summary.py"

  "analytics/feature_engineering/__init__.py"
  "analytics/feature_engineering/fe_planner.py"
  "analytics/feature_engineering/pipeline_builder.py"
  "analytics/feature_engineering/numeric_transformers.py"
  "analytics/feature_engineering/categorical_transformers.py"
  "analytics/feature_engineering/datetime_transformers.py"
  "analytics/feature_engineering/text_transformers.py"
  "analytics/feature_engineering/rare_label_handler.py"
  "analytics/feature_engineering/feature_selector.py"
  "analytics/feature_engineering/leakage_guard.py"
  "analytics/feature_engineering/feature_registry.py"
  "analytics/feature_engineering/fe_summary.py"

  "analytics/splitting/__init__.py"
  "analytics/splitting/split_router.py"
  "analytics/splitting/standard_split.py"
  "analytics/splitting/stratified_split.py"
  "analytics/splitting/time_split.py"
  "analytics/splitting/split_summary.py"

  "analytics/modeling/__init__.py"
  "analytics/modeling/task_detector.py"
  "analytics/modeling/model_selector.py"
  "analytics/modeling/baseline_trainer.py"
  "analytics/modeling/classifier_trainer.py"
  "analytics/modeling/regressor_trainer.py"
  "analytics/modeling/clustering_runner.py"
  "analytics/modeling/anomaly_model_runner.py"
  "analytics/modeling/model_registry.py"
  "analytics/modeling/training_log.py"

  "analytics/evaluation/__init__.py"
  "analytics/evaluation/evaluator.py"
  "analytics/evaluation/classification_metrics.py"
  "analytics/evaluation/regression_metrics.py"
  "analytics/evaluation/clustering_metrics.py"
  "analytics/evaluation/threshold_analysis.py"
  "analytics/evaluation/error_analysis.py"
  "analytics/evaluation/evaluation_summary.py"

  "analytics/explainability/__init__.py"
  "analytics/explainability/importance_explainer.py"
  "analytics/explainability/shap_explainer.py"
  "analytics/explainability/partial_dependence.py"
  "analytics/explainability/top_driver_summary.py"
  "analytics/explainability/explainability_summary.py"

  "reporting/__init__.py"
  "reporting/master_report_builder.py"
  "reporting/executive_summary_builder.py"
  "reporting/data_quality_section.py"
  "reporting/eda_section.py"
  "reporting/outlier_section.py"
  "reporting/feature_engineering_section.py"
  "reporting/model_section.py"
  "reporting/explainability_section.py"
  "reporting/insights_section.py"
  "reporting/critique_section.py"
  "reporting/recommendation_section.py"
  "reporting/markdown_report.py"
  "reporting/html_report.py"
  "reporting/report_exporter.py"

  "artifacts/__init__.py"
  "artifacts/artifact_manager.py"
  "artifacts/path_registry.py"
  "artifacts/dataset_exporter.py"
  "artifacts/chart_exporter.py"
  "artifacts/model_exporter.py"
  "artifacts/pipeline_exporter.py"
  "artifacts/metadata_exporter.py"
  "artifacts/run_manifest.py"

  "utils/__init__.py"
  "utils/logger.py"
  "utils/decorators.py"
  "utils/io_utils.py"
  "utils/dataframe_utils.py"
  "utils/type_utils.py"
  "utils/plotting.py"
  "utils/serialization.py"
  "utils/hashing.py"
  "utils/timing.py"
  "utils/safe_checks.py"

  "prompts/planner_prompt.txt"
  "prompts/insight_prompt.txt"
  "prompts/critic_prompt.txt"
  "prompts/recommendation_prompt.txt"
  "prompts/explanation_prompt.txt"

  "docs/architecture.md"
  "docs/pipeline.md"
  "docs/coding_rules.md"
  "docs/package_strategy.md"
  "docs/runbook.md"
  "docs/module_responsibility.md"

  "tests/__init__.py"
  "tests/conftest.py"
  "tests/unit/test_loader.py"
  "tests/unit/test_schema_detector.py"
  "tests/unit/test_profiler.py"
  "tests/unit/test_cleaner.py"
  "tests/unit/test_outlier_router.py"
  "tests/unit/test_feature_pipeline.py"
  "tests/unit/test_splitter.py"
  "tests/unit/test_model_selector.py"
  "tests/unit/test_evaluator.py"
  "tests/unit/test_agents_parser.py"

  "tests/integration/test_full_pipeline.py"
  "tests/integration/test_supervised_flow.py"
  "tests/integration/test_unsupervised_flow.py"
  "tests/integration/test_time_series_like_flow.py"
  "tests/integration/test_artifact_generation.py"

  "tests/smoke/test_streamlit_boot.py"
  "tests/smoke/test_template_generation.py"
  "tests/smoke/test_run_minimal_csv.py"

  "sample_data/tiny_classification.csv"
  "sample_data/tiny_regression.csv"
  "sample_data/messy_data_sample.csv"
)

for dir in "${DIRS[@]}"; do
  mkdir -p "${PROJECT_ROOT}/${dir}"
done

for file in "${FILES[@]}"; do
  touch "${PROJECT_ROOT}/${file}"
done

cat > README.md <<'EOF'
# Agentic CSV Data Scientist

An end-to-end agentic AI system for CSV analysis, EDA, feature engineering,
model training, explainability, dashboarding, and report generation.

## Initial bootstrap
1. Run template.bash
2. Create conda environment
3. Install requirements
4. Start implementing Reply 2 modules
EOF

cat > .env.example <<'EOF'
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
APP_ENV=development
EOF

cat > .gitignore <<'EOF'
__pycache__/
*.pyc
*.pyo
*.pyd
.python-version
.env
.venv/
venv/
.ipynb_checkpoints/
.pytest_cache/
.mypy_cache/
.coverage
htmlcov/
dist/
build/
*.egg-info/
outputs/reports/*
outputs/charts/*
outputs/datasets/*
outputs/models/*
outputs/pipelines/*
outputs/metadata/*
outputs/runs/*
!.gitkeep
EOF

cat > pyproject.toml <<'EOF'
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "agentic_csv_data_scientist"
version = "0.1.0"
description = "Agentic CSV Data Scientist"
readme = "README.md"
requires-python = ">=3.11"
EOF

cat > requirements.txt <<'EOF'
pandas
numpy
scikit-learn
streamlit
plotly
matplotlib
seaborn
scipy
joblib
pydantic
python-dotenv
duckdb
ydata-profiling
feature-engine
pyod
imbalanced-learn
category-encoders
shap
cleanlab
evidently
statsmodels
pytest
EOF

for keepdir in \
  outputs/reports \
  outputs/charts \
  outputs/datasets \
  outputs/models \
  outputs/pipelines \
  outputs/metadata \
  outputs/runs
do
  touch "${keepdir}/.gitkeep"
done

echo "[INFO] Scaffold created successfully."
echo "[INFO] Next steps:"
echo "       1) chmod +x template.bash"
echo "       2) ./template.bash"
echo "       3) create conda env"
echo "       4) pip install -r requirements.txt"