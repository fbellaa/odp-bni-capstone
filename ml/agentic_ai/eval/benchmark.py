from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..agent import QWEN_SYSTEM
from ..artifacts import ArtifactStore
from ..config import SETTINGS
from ..ollama_client import OllamaClient
from ..parser import QwenStructuredExtractor
from ..schemas import BorrowerExtraction, MODEL_KEYS, REQUIRED_ML_TOOLS
from ..tool_registry import all_definitions, dispatch


@dataclass
class ToolBenchmarkRun:
    model: str
    case_id: str
    expected_tools: list[str]
    actual_tools: list[str]
    tool_arguments: list[dict[str, Any]]
    error: str | None = None


def load_goldens(path: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(path) if path else Path(__file__).with_name("goldens.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def feature_catalog(store: ArtifactStore) -> dict[str, list[dict[str, Any]]]:
    return {
        key: [
            {"name": f.name, "dtype": f.dtype, "description": f.description}
            for f in store.feature_defs(key)
        ]
        for key in MODEL_KEYS
    }


def dummy_feature_context(store: ArtifactStore) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in MODEL_KEYS:
        values: dict[str, Any] = {}
        for f in store.feature_defs(key):
            t = f.json_type()
            if t == "boolean":
                values[f.name] = False
            elif t == "integer":
                values[f.name] = 1
            elif t == "string":
                values[f.name] = "UNKNOWN"
            else:
                values[f.name] = 1.0
        out[key] = {"features": values, "missing_feature_names": []}
    return out


def expected_tool_arguments(context: dict[str, Any]) -> list[dict[str, Any]]:
    tool_to_model = {
        "predict_pd": "pd",
        "predict_ews": "ews",
        "predict_lgd": "lgd",
        "predict_pd_cluster": "pd_cluster",
    }
    return [{"features": context[tool_to_model[t]]["features"]} for t in REQUIRED_ML_TOOLS]


def run_extractor(case: dict[str, Any], *, model: str) -> BorrowerExtraction:
    store = ArtifactStore()
    client = OllamaClient()
    return QwenStructuredExtractor(client=client).extract(
        case["document_text"],
        feature_catalog=feature_catalog(store),
        manual_input=case.get("manual_input"),
        model_override=model,
    )


def run_tool_caller(*, model: str, case_id: str = "routing") -> ToolBenchmarkRun:
    store = ArtifactStore()
    client = OllamaClient()
    context = dummy_feature_context(store)
    payload = {
        "assessment_contract": {"required_tools": list(REQUIRED_ML_TOOLS)},
        "features_by_model": context,
    }
    messages = [
        {"role": "system", "content": QWEN_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    names: list[str] = []
    args_seen: list[dict[str, Any]] = []
    try:
        for _ in range(4):
            reply = client.chat(model=model, messages=messages, tools=all_definitions(), temperature=0)
            messages.append(reply)
            calls = reply.get("tool_calls") or []
            if not calls:
                if set(names) >= set(REQUIRED_ML_TOOLS):
                    break
                messages.append({"role": "user", "content": "Call every remaining mandatory tool now."})
                continue
            for call in calls:
                fn = call.get("function", {})
                name = str(fn.get("name", ""))
                raw = fn.get("arguments") or {}
                args = raw if isinstance(raw, dict) else json.loads(raw)
                names.append(name)
                args_seen.append(args)
                trace = dispatch(name, args, execute=False)
                messages.append({"role": "tool", "tool_name": name, "content": trace.model_content()})
            if set(names) >= set(REQUIRED_ML_TOOLS):
                break
        return ToolBenchmarkRun(model, case_id, list(REQUIRED_ML_TOOLS), names, args_seen, None)
    except Exception as exc:
        return ToolBenchmarkRun(model, case_id, list(REQUIRED_ML_TOOLS), names, args_seen, str(exc))
