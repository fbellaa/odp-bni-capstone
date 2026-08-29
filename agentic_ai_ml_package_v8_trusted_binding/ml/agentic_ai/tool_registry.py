from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .artifacts import ArtifactError, ArtifactStore, ToolInputError
from .schemas import MODEL_KEYS, REQUIRED_ML_TOOLS

STORE = ArtifactStore()

TOOL_TO_MODEL = {
    "predict_pd": "pd",
    "predict_ews": "ews",
    "predict_lgd": "lgd",
    "predict_pd_cluster": "pd_cluster",
}

MODEL_TO_TOOL = {v: k for k, v in TOOL_TO_MODEL.items()}

TOOL_DESCRIPTION = {
    "predict_pd": (
        "Run the saved 12-month Probability of Default model for the CURRENT borrower. "
        "Do not provide model features. Python binds the verified PD feature payload."
    ),
    "predict_ews": (
        "Run the saved Early Warning System model for the CURRENT borrower. "
        "Do not provide model features. Python binds the verified EWS feature payload."
    ),
    "predict_lgd": (
        "Run the saved Loss Given Default model for the CURRENT borrower. "
        "Do not provide model features. Python binds the verified LGD feature payload."
    ),
    "predict_pd_cluster": (
        "Run the saved PD borrower clustering model for the CURRENT borrower. "
        "Do not provide model features. Python binds the verified clustering feature payload."
    ),
}


@dataclass
class ToolTrace:
    name: str

    # arguments = TRUSTED arguments actually executed by Python.
    arguments: dict[str, Any]

    result: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: int = 0
    duplicate_blocked: bool = False
    caller: str = "unknown"

    # Raw arguments proposed by the LLM before trusted binding.
    # V8 Qwen tools are parameterless, so this should normally be {}.
    llm_arguments: dict[str, Any] | None = None

    # Audit trail showing where execution arguments came from.
    binding_source: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def llm_argument_policy_compliant(self) -> bool:
        return not bool(self.llm_arguments or {})

    def model_content(self) -> str:
        if self.error:
            return json.dumps({"error": self.error}, ensure_ascii=False)
        return json.dumps(self.result, ensure_ascii=False, default=str)


@dataclass
class ToolRecord:
    traces: list[ToolTrace] = field(default_factory=list)

    def add(self, trace: ToolTrace) -> ToolTrace:
        self.traces.append(trace)
        return trace

    @property
    def attempted_names(self) -> set[str]:
        return {
            t.name
            for t in self.traces
            if t.name in TOOL_TO_MODEL and not t.duplicate_blocked
        }

    @property
    def missing_required_tools(self) -> set[str]:
        return set(REQUIRED_ML_TOOLS) - self.attempted_names

    @property
    def coverage(self) -> float:
        return (
            len(self.attempted_names & set(REQUIRED_ML_TOOLS))
            / len(REQUIRED_ML_TOOLS)
        )

    @property
    def qwen_attempted_names(self) -> set[str]:
        return {
            t.name
            for t in self.traces
            if (
                t.name in TOOL_TO_MODEL
                and not t.duplicate_blocked
                and t.caller == "qwen"
            )
        }

    @property
    def qwen_coverage(self) -> float:
        return (
            len(self.qwen_attempted_names & set(REQUIRED_ML_TOOLS))
            / len(REQUIRED_ML_TOOLS)
        )

    @property
    def qwen_argument_policy_compliance(self) -> float:
        traces = [
            t for t in self.traces
            if t.caller == "qwen"
            and t.name in TOOL_TO_MODEL
            and not t.duplicate_blocked
        ]
        if not traces:
            return 0.0
        return sum(t.llm_argument_policy_compliant for t in traces) / len(traces)

    @property
    def last_success_by_name(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for trace in self.traces:
            if trace.success and trace.result is not None and not trace.duplicate_blocked:
                out[trace.name] = trace.result
        return out


def _tool_function(model_key: str) -> Callable[..., dict[str, Any]]:
    def run(*, features: dict[str, Any]) -> dict[str, Any]:
        """Execute a model with Python-verified borrower features.

        Missing fields are NOT invented. ArtifactStore builds the full model frame
        and saved preprocessing/imputers handle unavailable columns where supported.
        """
        if not isinstance(features, dict):
            raise ToolInputError("features harus berupa JSON object/dict.")

        expected = STORE.feature_names(model_key)
        clean = {name: features[name] for name in expected if name in features}
        result = STORE.predict(model_key, clean)

        quality = result.get("input_quality", {})
        missing_n = int(quality.get("missing_feature_count", 0) or 0)
        status = "scored_with_imputation" if missing_n else "scored"
        return {"status": status, **result}

    return run


PETA: dict[str, Callable[..., dict[str, Any]]] = {
    tool: _tool_function(model_key)
    for tool, model_key in TOOL_TO_MODEL.items()
}


def definition(tool_name: str) -> dict[str, Any]:
    """Qwen sees tool-selection only.

    Business/model features are deliberately absent from the function schema.
    The Python orchestrator binds the trusted payload after Qwen selects a tool.
    """
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": TOOL_DESCRIPTION[tool_name],
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }


def all_definitions() -> list[dict[str, Any]]:
    return [definition(name) for name in REQUIRED_ML_TOOLS]


def relevant_model_keys() -> list[str]:
    return list(MODEL_KEYS)


def dispatch(
    name: str,
    arguments: dict[str, Any],
    *,
    execute: bool = True,
    caller: str = "unknown",
    llm_arguments: dict[str, Any] | None = None,
    binding_source: str | None = None,
) -> ToolTrace:
    start = time.perf_counter()
    fn = PETA.get(name)

    if fn is None:
        return ToolTrace(
            name=name,
            arguments=arguments or {},
            error=f"Unknown tool {name!r}. Available: {', '.join(sorted(PETA))}",
            duration_ms=0,
            caller=caller,
            llm_arguments=llm_arguments,
            binding_source=binding_source,
        )

    try:
        if execute:
            result = fn(**(arguments or {}))
        else:
            result = {
                "status": "mocked",
                "tool": name,
                "arguments_received": arguments or {},
            }
        error = None
    except (ToolInputError, ArtifactError, TypeError, ValueError) as exc:
        result, error = None, str(exc)
    except Exception as exc:
        result, error = None, f"Unexpected tool error: {exc}"

    return ToolTrace(
        name=name,
        arguments=arguments or {},
        result=result,
        error=error,
        duration_ms=int((time.perf_counter() - start) * 1000),
        caller=caller,
        llm_arguments=llm_arguments,
        binding_source=binding_source,
    )
