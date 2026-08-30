from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .artifacts import ArtifactError, ArtifactStore, ToolInputError
from .rag_adapter import PolicyRAGError, query_credit_policy
from .schemas import (
    MODEL_KEYS,
    POLICY_RAG_TOOL,
    REQUIRED_AGENT_TOOLS,
    REQUIRED_ML_TOOLS,
)

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
    POLICY_RAG_TOOL: (
        "Search the commercial-credit POLICY RAG corpus and return a grounded Indonesian "
        "policy answer with article/page citations. Use this only after the ML tools have "
        "been attempted. The query should describe the policy issue you need to verify; "
        "do not invent article numbers."
    ),
}


@dataclass
class ToolTrace:
    name: str

    # arguments = arguments actually executed by Python.
    arguments: dict[str, Any]

    result: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: int = 0
    duplicate_blocked: bool = False
    caller: str = "unknown"

    # Raw arguments proposed by Qwen before any trusted ML binding.
    llm_arguments: dict[str, Any] | None = None

    # e.g. python_feature_context or qwen_policy_query
    binding_source: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def llm_argument_policy_compliant(self) -> bool:
        # RAG is intentionally allowed to carry a natural-language query.
        if self.name == POLICY_RAG_TOOL:
            return True

        # V9.2 ML contract: exactly {"run": true}.
        args = self.llm_arguments or {}
        return (
            set(args.keys()) == {"run"}
            and args.get("run") is True
        )

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
        """Backward-compatible ML attempted names."""
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
        """Backward-compatible ML coverage."""
        return (
            len(self.attempted_names & set(REQUIRED_ML_TOOLS))
            / len(REQUIRED_ML_TOOLS)
        )

    @property
    def qwen_attempted_names(self) -> set[str]:
        """Backward-compatible Qwen ML attempted names."""
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
        """Backward-compatible Qwen ML coverage."""
        return (
            len(self.qwen_attempted_names & set(REQUIRED_ML_TOOLS))
            / len(REQUIRED_ML_TOOLS)
        )

    @property
    def qwen_agent_attempted_names(self) -> set[str]:
        return {
            t.name
            for t in self.traces
            if (
                t.name in REQUIRED_AGENT_TOOLS
                and not t.duplicate_blocked
                and t.caller == "qwen"
            )
        }

    @property
    def qwen_agent_tool_coverage(self) -> float:
        return (
            len(self.qwen_agent_attempted_names & set(REQUIRED_AGENT_TOOLS))
            / len(REQUIRED_AGENT_TOOLS)
        )

    @property
    def rag_attempted(self) -> bool:
        return any(
            t.name == POLICY_RAG_TOOL
            and not t.duplicate_blocked
            for t in self.traces
        )

    @property
    def rag_qwen_attempted(self) -> bool:
        return any(
            t.name == POLICY_RAG_TOOL
            and t.caller == "qwen"
            and not t.duplicate_blocked
            for t in self.traces
        )

    @property
    def rag_result(self) -> dict[str, Any] | None:
        for trace in reversed(self.traces):
            if (
                trace.name == POLICY_RAG_TOOL
                and trace.success
                and trace.result is not None
                and not trace.duplicate_blocked
            ):
                return trace.result
        return None

    @property
    def rag_retrieved(self) -> bool:
        result = self.rag_result or {}
        return result.get("status") == "retrieved"

    @property
    def qwen_argument_policy_compliance(self) -> float:
        traces = [
            t for t in self.traces
            if (
                t.caller == "qwen"
                and t.name in TOOL_TO_MODEL
                and not t.duplicate_blocked
            )
        ]
        if not traces:
            return 0.0
        return sum(t.llm_argument_policy_compliant for t in traces) / len(traces)

    @property
    def last_success_by_name(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for trace in self.traces:
            if (
                trace.success
                and trace.result is not None
                and not trace.duplicate_blocked
            ):
                out[trace.name] = trace.result
        return out


def _tool_function(model_key: str) -> Callable[..., dict[str, Any]]:
    def run(*, features: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(features, dict):
            raise ToolInputError("features harus berupa JSON object/dict.")

        expected = STORE.feature_names(model_key)
        clean = {
            name: features[name]
            for name in expected
            if name in features
        }
        result = STORE.predict(model_key, clean)

        quality = result.get("input_quality", {})
        missing_n = int(
            quality.get("missing_feature_count", 0)
            or 0
        )
        status = (
            "scored_with_imputation"
            if missing_n
            else "scored"
        )
        return {"status": status, **result}

    return run


def _rag_tool(
    *,
    query: str,
    top_k: int | None = None,
) -> dict[str, Any]:
    if top_k is not None:
        top_k = max(1, min(int(top_k), 10))
    return query_credit_policy(
        query,
        top_k=top_k,
    )


PETA: dict[str, Callable[..., dict[str, Any]]] = {
    **{
        tool: _tool_function(model_key)
        for tool, model_key in TOOL_TO_MODEL.items()
    },
    POLICY_RAG_TOOL: _rag_tool,
}


def ml_definition(tool_name: str) -> dict[str, Any]:
    """ML tools use one explicit trigger argument for stable Ollama tool calling.

    Qwen must call each ML tool with exactly:
        {"run": true}

    The `run` field is NEVER forwarded to the model. Python discards it and
    binds the exact trusted `feature_context` for the selected model.
    """
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": (
                TOOL_DESCRIPTION[tool_name]
                + ' Call this tool with exactly {"run": true}. '
                "Do not provide model features."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "run": {
                        "type": "boolean",
                        "description": (
                            "Set to true to run this model for the current borrower. "
                            "This is only a tool-call trigger; Python supplies all ML features."
                        ),
                    }
                },
                "required": ["run"],
                "additionalProperties": False,
            },
        },
    }


def rag_definition() -> dict[str, Any]:
    """RAG query is intentionally generated by Qwen."""
    return {
        "type": "function",
        "function": {
            "name": POLICY_RAG_TOOL,
            "description": TOOL_DESCRIPTION[POLICY_RAG_TOOL],
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A concise Indonesian policy question grounded in the "
                            "borrower/application and verified ML results."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Number of policy chunks to retrieve.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }


def definition(tool_name: str) -> dict[str, Any]:
    if tool_name == POLICY_RAG_TOOL:
        return rag_definition()
    return ml_definition(tool_name)


def ml_definitions() -> list[dict[str, Any]]:
    return [
        ml_definition(name)
        for name in REQUIRED_ML_TOOLS
    ]


def all_definitions() -> list[dict[str, Any]]:
    return ml_definitions() + [rag_definition()]


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
            error=(
                f"Unknown tool {name!r}. "
                f"Available: {', '.join(sorted(PETA))}"
            ),
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
    except (
        ToolInputError,
        ArtifactError,
        PolicyRAGError,
        TypeError,
        ValueError,
    ) as exc:
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
