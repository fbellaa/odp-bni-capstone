# CHANGELOG V3 — Colab Ready

- Added `05_agentic_ai_end_to_end_colab.ipynb`.
- Added Colab GPU/runtime checks.
- Added optional GitHub clone + package overlay workflow.
- Added automated installation of AI and model-specific requirements.
- Added local Ollama installation/startup for Colab.
- Added `OLLAMA_MAX_LOADED_MODELS=1` / `OLLAMA_NUM_PARALLEL=1` setup.
- Added configurable `AI_OLLAMA_KEEP_ALIVE`; Colab notebook sets it to `0` to unload between SahabatAI → Qwen → SahabatAI stages.
- Added artifact presence checks and champion load checks.
- Added Colab file uploader for RM documents.
- Added end-to-end run, verified tool trace, feature provenance, and DeepEval sections.
- SahabatAI model tag remains explicit/configurable because the exact Ollama-compatible GGUF quantization must be chosen and documented by the project.
