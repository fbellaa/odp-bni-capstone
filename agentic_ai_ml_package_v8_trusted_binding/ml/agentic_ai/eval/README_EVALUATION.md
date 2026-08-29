# V7.1 Evaluation — DeepEval

V7.1 evaluates the architecture **per layer** instead of forcing one opaque score.

## Layer 1 — Document mapper

Deterministic exact-match benchmark against `goldens_v7.jsonl`.

Metrics:
- field recall
- field precision
- field F1
- forbidden / hallucinated field hits

This layer intentionally does **not** use an LLM judge because numeric extraction
from a known document should be evaluated deterministically.

## Layer 2 — Qwen tool calling

Uses DeepEval `ToolCorrectnessMetric`.

The expected contract is exactly four Qwen calls:

- `predict_pd`
- `predict_ews`
- `predict_lgd`
- `predict_pd_cluster`

Evaluation includes exact input parameters, Qwen coverage, duplicate calls,
invalid calls, and exact argument fidelity.

Runtime model failure is separated from tool-calling correctness. For example,
Qwen can correctly call `predict_lgd` even if an incompatible LGD artifact later
throws an sklearn runtime error.

## Layer 3 — SahabatAI narrator

Uses DeepEval `GEval` with a local Ollama judge.

The judge checks:
- numeric faithfulness to verified ML outputs
- no invented model result
- failed models are not presented as successful
- imputed/missing inputs are not presented as observed facts
- no unsupported causal claims

Set `DEEPEVAL_JUDGE_MODEL` if you want a separate local judge model.
If unset, V7.1 uses the configured Qwen agent model as the local judge.

## Layer 4 — End-to-end scorecard

Reports:
- Qwen tool coverage
- successful model count
- model success rate
- feature completeness by model
- final answer availability
- stop reason

The system deliberately does not average unrelated metrics into one arbitrary
weighted number.

## Recommended capstone benchmark

For a defensible model/agent evaluation, expand `goldens_v7.jsonl` to 30–100 cases:
formal Indonesian, slang/abbreviations, typos, multiple documents, missing data,
contradictions, tables, scanned documents, and irrelevant pages.
