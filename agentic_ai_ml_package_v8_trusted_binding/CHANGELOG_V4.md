# CHANGELOG V4

## Main revision

V4 supports the workflow where the ML model artifacts are still stored only on the user's local laptop and are not committed to GitHub.

### Changed
- Repository is still cloned from GitHub for source code.
- New Colab Step 1B uploads a single `artifacts.zip` from the user's laptop.
- ZIP extraction auto-detects `artifacts/` or `ml/artifacts/`.
- Uploaded artifacts are copied to `/content/odp-bni-capstone/ml/artifacts`.
- Artifact upload occurs before model-specific dependency installation.
- Preflight now explicitly validates runtime-uploaded artifacts.
- Added optional note for persisting `artifacts.zip` in Google Drive.
- Canonical notebook now points to the V4 Colab workflow.

### Unchanged
- SahabatAI extracts borrower facts.
- Feature engineering is deterministic Python.
- Qwen is the mandatory ML tool orchestrator.
- PD, EWS, LGD, and PD Cluster are all attempted.
- Verified tool results are narrated by SahabatAI.
