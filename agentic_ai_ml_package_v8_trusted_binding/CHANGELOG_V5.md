# CHANGELOG V5

## Why V5 exists

The earlier package stopped PD/EWS/LGD/PD Cluster whenever any expected feature was
missing. That is too strict for the actual saved modeling pipelines, because the
supervised models were trained with imputers.

## Changes

- All four ML tools are still mandatory.
- Missing expected model columns are padded with `NaN`, not invented values.
- Saved model preprocessing/imputer owns median/most-frequent imputation.
- Tool output adds `input_quality` and `status=scored_with_imputation` when relevant.
- Narrator must report data completeness rather than claiming the model could not run.
- Exact prefixed model features are mapped from canonical extracted facts.
- Added safe deterministic derived features (only formulas explicitly allowed in code).
- PD calibrator now receives `logit(raw_probability)`, matching the PD training notebook.
- PD Cluster metadata keys `model`/`model_version` are no longer misread as features.
- PD Cluster scoring supports the saved training sequence: rating map → numeric coercion
  → winsor bounds → median imputer → scaler → PCA → KMeans.
- Added a clear runtime error if the cluster artifact lacks the preprocessing objects.
- Added multimodal document extraction helper: native text → Tesseract OCR → optional VLM.
- Colab notebook updated for Python 3.13 dependency compatibility and manual Ollama install.
