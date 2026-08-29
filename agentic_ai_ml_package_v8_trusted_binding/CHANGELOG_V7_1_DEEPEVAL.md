# CHANGELOG V7.1 — DeepEval

V7 architecture is unchanged. This release adds layer-wise evaluation.

## Added

- Deterministic document-mapper golden benchmark.
- `goldens_v7.jsonl` covering:
  - financial statement fields,
  - application fields,
  - EWS behavior,
  - cash-flow field,
  - irrelevant-text hallucination check.
- DeepEval `ToolCorrectnessMetric` for Qwen tool calling.
- Exact Qwen tool-argument fidelity checks.
- Duplicate and invalid tool-call diagnostics.
- DeepEval `GEval` for SahabatAI narrator groundedness.
- End-to-end evaluation scorecard.
- JSON evaluation report export from Colab.
- Notebook evaluation section 17A–17F.

## Important interpretation

Tool-calling correctness and ML runtime success are intentionally separated.

Example:
- Qwen calls `predict_lgd` correctly -> Qwen tool-calling can pass.
- LGD sklearn artifact crashes afterwards -> LGD runtime is reported as failed.

This prevents a model-artifact compatibility problem from being incorrectly
reported as an agent/tool-calling failure.
