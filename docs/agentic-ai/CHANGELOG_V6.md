# CHANGELOG V6

## Architecture

V6 changes the LLM assignment to:

- native PyPDF/DOCX for machine-readable documents;
- Qwen3-VL 4B for visual fallback;
- Qwen2.5 7B for structured borrower fact extraction;
- deterministic Python for feature engineering;
- Qwen2.5 7B for mandatory ML tool orchestration;
- SahabatAI only for final Indonesian narration.

## Performance

- Added `document_reducer.py` and fast extraction mode.
- Long documents are reduced to relevant, source-tagged excerpts before one structured Qwen extraction call.
- Removes the need for many SahabatAI chunk calls in the default workflow.

## Reliability

- Agent completion now depends on successful tool results, not only attempted tool names.
- Python fallback deterministically attempts any mandatory tool that Qwen omitted or called incorrectly.
- Narrator receives only verified successful results plus a separate error list.
- Missing features remain `NaN` and are handled by saved model preprocessing where supported.

## Model defaults

- Qwen extractor: `qwen2.5:7b-instruct`
- Qwen agent: `qwen2.5:7b-instruct`
- VLM: `qwen3-vl:4b-instruct`
- SahabatAI narrator: supplied through `SAHABAT_MODEL`.
