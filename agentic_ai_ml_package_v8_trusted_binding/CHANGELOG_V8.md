# CHANGELOG V8 — Trusted Python Feature Binding

## Main architectural change

Qwen remains the mandatory tool-calling agent, but Qwen no longer receives,
copies, or generates ML feature payloads.

### New execution contract

`Documents → deterministic mapping → deterministic feature engineering → Qwen selects tool → Python binds exact trusted features → ML model`

### Why

V7 debugging showed Qwen could select the correct tools but hallucinate or mix
large model feature payloads. V8 removes that failure mode by design.

### Added

- Parameterless Qwen ML tool schemas.
- Python trusted feature binding after tool selection.
- `ToolTrace.llm_arguments` audit field.
- `ToolTrace.binding_source`.
- Qwen argument-policy compliance metric.
- Python binding-fidelity metric.
- DeepEval now evaluates Qwen tool selection separately from Python payload binding.
- DeepEval local Ollama dependency included in runtime requirements.
- Narrator GEval evaluates factual grounding only; different narrative formatting is not penalized.
- Revenue mapping no longer mistakes `Total Pendapatan Lain-Lain` for core sales/revenue.
- Conservative categorical cleanup for KBLI/facility fields.

### Runtime interpretation

A result can legitimately be:

- Qwen tool coverage: 100%
- Python feature binding fidelity: 100%
- ML runtime: 3/4 if LGD artifact remains incompatible

That means orchestration is correct while one model artifact still has a runtime issue.
