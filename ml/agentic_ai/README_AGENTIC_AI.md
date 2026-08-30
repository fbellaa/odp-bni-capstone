# V7.1 DeepEval Note

The V7 architecture is unchanged. V7.1 adds layer-wise evaluation:
document mapping, Qwen tool calling, Qwen grounded narration, and
an end-to-end scorecard.

Use:
`ml/notebooks/05_agentic_ai_end_to_end_v7_deepeval_colab.ipynb`

Evaluation implementation:
`ml/agentic_ai/eval/deepeval_v7.py`

See:
`ml/agentic_ai/eval/README_EVALUATION.md`

---

# Agentic AI Credit Risk — V7

V7 removes LLM mapping as the primary bottleneck.

## Architecture

```text
RM uploads multiple documents
        |
        v
Native PDF/DOCX extraction
OCR / Qwen3-VL only when needed
        |
        v
DeterministicDocumentMapper
- Indonesian accounting labels -> canonical raw facts
- explicit unit normalization
- provenance per fact
        |
        v
Deterministic FeatureEngineer
- exact aliases to model columns
- audited formulas only
        |
        v
Qwen2.5 Tool-Calling Agent
- predict_pd
- predict_ews
- predict_lgd
- predict_pd_cluster
        |
        v
Verified ML outputs
        |
        v
Qwen Indonesian credit-risk narrator
```

Qwen is still the agent/tool orchestrator. It is simply no longer trusted to do the
primary financial-label mapping that deterministic Python can do faster and more reliably.

## Optional Qwen semantic fallback

Set:

```bash
AI_QWEN_SEMANTIC_FALLBACK=1
```

Only use this when a document uses unfamiliar wording. Deterministic facts always win
when the fallback disagrees.

## Important

The deterministic formula registry must remain aligned with the training feature
engineering. Features whose training formula has not been verified should remain missing
and be handled by the saved model imputer rather than invented.
