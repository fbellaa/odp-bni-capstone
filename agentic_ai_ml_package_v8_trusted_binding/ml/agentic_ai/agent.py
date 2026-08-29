from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .config import SETTINGS, Settings
from .ollama_client import OllamaClient, OllamaError
from .schemas import REQUIRED_ML_TOOLS
from .tool_registry import (
    TOOL_TO_MODEL,
    ToolRecord,
    ToolTrace,
    all_definitions,
    dispatch,
)

QWEN_SYSTEM = """\
You are the ML TOOL ORCHESTRATOR for a commercial credit-risk system.

YOUR ROLE IS TOOL SELECTION ONLY.

MANDATORY CONTRACT FOR EVERY BORROWER:
1. Call exactly these four tools:
   - predict_pd
   - predict_ews
   - predict_lgd
   - predict_pd_cluster
2. Do NOT calculate, copy, generate, rename, transform, or supply model features.
3. Do NOT supply business arguments to the tools. Call each tool with an empty JSON object.
4. Python securely binds the verified borrower feature payload after you select a tool.
5. Tool outputs are authoritative.
6. A model may fail at runtime; that is a model/artifact issue, not permission to invent a result.

You are evaluated on selecting the correct tools, not on reproducing model inputs.
"""


@dataclass
class AgentResult:
    record: ToolRecord = field(default_factory=ToolRecord)
    messages: list[dict[str, Any]] = field(default_factory=list)
    final_content: str = ""
    rounds: int = 0
    stopped_reason: str = "finished"

    @property
    def complete(self) -> bool:
        return set(REQUIRED_ML_TOOLS).issubset(
            self.record.last_success_by_name
        )


class QwenMLAgent:
    def __init__(
        self,
        client: OllamaClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.s = settings or SETTINGS
        self.client = client or OllamaClient(self.s)

    @staticmethod
    def _arguments(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                x = json.loads(raw)
                return x if isinstance(x, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def _trusted_args(
        tool_name: str,
        feature_context: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        model_key = TOOL_TO_MODEL[tool_name]
        features = (
            feature_context
            .get(model_key, {})
            .get("features", {})
        )
        return {
            "features": dict(features)
            if isinstance(features, dict)
            else {}
        }

    @staticmethod
    def _agent_visible_summary(
        feature_context: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Give Qwen availability metadata, never business feature values."""
        summary: dict[str, dict[str, Any]] = {}
        for model_key, ctx in feature_context.items():
            features = ctx.get("features", {})
            missing = ctx.get("missing_feature_names", [])
            summary[model_key] = {
                "available_feature_count": len(features) if isinstance(features, dict) else 0,
                "missing_feature_count": len(missing) if isinstance(missing, list) else 0,
                "model_must_be_attempted": True,
            }
        return summary

    def _python_fallback(
        self,
        record: ToolRecord,
        messages: list[dict[str, Any]],
        feature_context: dict[str, dict[str, Any]],
        *,
        execute_tools: bool,
    ) -> None:
        successful = set(record.last_success_by_name)

        for tool_name in REQUIRED_ML_TOOLS:
            if tool_name in successful:
                continue

            trusted_args = self._trusted_args(
                tool_name,
                feature_context,
            )

            trace = dispatch(
                tool_name,
                trusted_args,
                execute=execute_tools,
                caller="python_fallback",
                llm_arguments=None,
                binding_source="python_feature_context",
            )

            record.add(trace)

            messages.append({
                "role": "tool",
                "tool_name": tool_name,
                "content": trace.model_content(),
            })

    def run(
        self,
        feature_context: dict[str, dict[str, Any]],
        *,
        execute_tools: bool = True,
        model_override: str | None = None,
    ) -> AgentResult:
        model = model_override or self.s.qwen_agent_model

        # IMPORTANT:
        # Feature values are intentionally NOT exposed to Qwen.
        payload = {
            "assessment_contract": {
                "required_tools": list(REQUIRED_ML_TOOLS),
                "required_tool_count": len(REQUIRED_ML_TOOLS),
                "tool_arguments_policy": "empty_object_only",
                "python_trusted_feature_binding": True,
            },
            "model_availability": self._agent_visible_summary(
                feature_context
            ),
        }

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": QWEN_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    payload,
                    ensure_ascii=False,
                    default=str,
                ),
            },
        ]

        record = ToolRecord()
        attempted: set[str] = set()
        last_round = 0

        for round_no in range(
            1,
            self.s.max_tool_rounds + 1,
        ):
            last_round = round_no

            try:
                reply = self.client.chat(
                    model=model,
                    messages=messages,
                    tools=all_definitions(),
                    temperature=self.s.agent_temperature,
                )
            except OllamaError as exc:
                messages.append({
                    "role": "user",
                    "content": f"Qwen runtime error: {exc}",
                })
                break

            messages.append(reply)
            calls = reply.get("tool_calls") or []

            for call in calls:
                fn = call.get("function", {})
                name = str(fn.get("name", ""))
                llm_args = self._arguments(
                    fn.get("arguments")
                )

                if name in attempted:
                    trace = ToolTrace(
                        name=name,
                        arguments={},
                        error=(
                            "Duplicate prediction tool call "
                            "blocked by orchestrator."
                        ),
                        duplicate_blocked=True,
                        caller="qwen",
                        llm_arguments=llm_args,
                        binding_source=None,
                    )

                elif name in TOOL_TO_MODEL:
                    # CRITICAL V8 GUARDRAIL:
                    # Ignore any feature values proposed by Qwen.
                    # Bind exact Python-generated borrower features.
                    trusted_args = self._trusted_args(
                        name,
                        feature_context,
                    )

                    trace = dispatch(
                        name,
                        trusted_args,
                        execute=execute_tools,
                        caller="qwen",
                        llm_arguments=llm_args,
                        binding_source="python_feature_context",
                    )

                    attempted.add(name)

                else:
                    trace = dispatch(
                        name,
                        llm_args,
                        execute=execute_tools,
                        caller="qwen",
                        llm_arguments=llm_args,
                        binding_source=None,
                    )

                record.add(trace)

                messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "content": trace.model_content(),
                })

            successful = set(
                record.last_success_by_name
            )

            if set(REQUIRED_ML_TOOLS).issubset(
                successful
            ):
                return AgentResult(
                    record=record,
                    messages=messages,
                    final_content=(
                        "Mandatory four-model assessment "
                        "completed by Qwen with trusted "
                        "Python feature binding."
                    ),
                    rounds=round_no,
                    stopped_reason="all_required_tools_successful",
                )

            not_attempted = sorted(
                set(REQUIRED_ML_TOOLS) - attempted
            )

            if not_attempted:
                messages.append({
                    "role": "user",
                    "content": (
                        "ASSESSMENT INCOMPLETE. Call these "
                        "remaining tools with EMPTY arguments: "
                        + ", ".join(not_attempted)
                    ),
                })
            else:
                # All tools were selected by Qwen. Remaining failures
                # are runtime/model failures, not missing orchestration.
                break

        # Reliability fallback remains available if Qwen omits a tool.
        self._python_fallback(
            record,
            messages,
            feature_context,
            execute_tools=execute_tools,
        )

        failed = sorted(
            set(REQUIRED_ML_TOOLS)
            - set(record.last_success_by_name)
        )

        if failed:
            return AgentResult(
                record=record,
                messages=messages,
                final_content=(
                    "Runtime failures remain for: "
                    + ", ".join(failed)
                ),
                rounds=last_round,
                stopped_reason="model_runtime_failure",
            )

        return AgentResult(
            record=record,
            messages=messages,
            final_content=(
                "Mandatory four-model assessment completed "
                "with trusted Python feature binding and "
                "Python verification."
            ),
            rounds=last_round,
            stopped_reason=(
                "all_required_tools_successful_with_fallback"
            ),
        )
