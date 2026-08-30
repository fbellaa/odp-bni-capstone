from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .config import SETTINGS, Settings
from .ollama_client import OllamaClient, OllamaError
from .schemas import (
    POLICY_RAG_TOOL,
    REQUIRED_AGENT_TOOLS,
    REQUIRED_ML_TOOLS,
)
from .tool_registry import (
    TOOL_TO_MODEL,
    ToolRecord,
    ToolTrace,
    dispatch,
    ml_definitions,
    rag_definition,
)

QWEN_SYSTEM = """\
You are the ML TOOL ORCHESTRATOR for a commercial credit-risk system.

PHASE 1 — ML TOOL SELECTION.

MANDATORY CONTRACT:
1. Call exactly these four ML tools:
   - predict_pd
   - predict_ews
   - predict_lgd
   - predict_pd_cluster
2. Do NOT calculate, copy, generate, rename, transform, or supply model features.
3. Call every ML tool with exactly: {"run": true}
4. The `run` argument is only a trigger and is not forwarded to the ML model.
5. Python securely binds the verified borrower feature payload after you select a tool.
6. Tool outputs are authoritative.
7. A model may fail at runtime; never invent a replacement result.

You are evaluated on selecting the correct tools, not on reproducing model inputs.
"""

POLICY_SYSTEM = """\
You are the POLICY RAG TOOL ORCHESTRATOR for a commercial credit-risk system.

PHASE 2 — POLICY RETRIEVAL.

You have already received the verified ML results and a small policy-relevant
borrower/application context.

Call `query_credit_policy` EXACTLY ONCE.

Rules:
1. Write one concise Indonesian policy query that would help an analyst/RM verify
   applicable credit-policy requirements for this case.
2. Base the query only on the supplied context and verified tool results.
3. Do not invent regulation names, article numbers, thresholds, or policy conclusions.
4. `query_credit_policy` is the only tool available in this phase.
5. The RAG tool itself is responsible for retrieving policy passages and citations.
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

    @property
    def policy_result(self) -> dict[str, Any] | None:
        return self.record.rag_result


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
            "features": (
                dict(features)
                if isinstance(features, dict)
                else {}
            )
        }

    @staticmethod
    def _agent_visible_summary(
        feature_context: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        summary: dict[str, dict[str, Any]] = {}
        for model_key, ctx in feature_context.items():
            features = ctx.get("features", {})
            missing = ctx.get("missing_feature_names", [])
            summary[model_key] = {
                "available_feature_count": (
                    len(features)
                    if isinstance(features, dict)
                    else 0
                ),
                "missing_feature_count": (
                    len(missing)
                    if isinstance(missing, list)
                    else 0
                ),
                "model_must_be_attempted": True,
            }
        return summary


    def _qwen_recover_missing_ml_tools(
        self,
        *,
        record: ToolRecord,
        messages: list[dict[str, Any]],
        feature_context: dict[str, dict[str, Any]],
        execute_tools: bool,
        model: str,
    ) -> int:
        """Ask Qwen again, one missing ML tool at a time.

        The initial phase still exposes all four tools together. If Qwen omits
        one, recovery exposes only that missing tool and requires a real
        ``{"run": true}`` function call. Python still binds the trusted feature
        payload after Qwen emits the call.
        """
        recovery_rounds = 0

        for tool_name in REQUIRED_ML_TOOLS:
            if tool_name in record.qwen_attempted_names:
                continue

            specs = [
                spec
                for spec in ml_definitions()
                if spec.get("function", {}).get("name") == tool_name
            ]

            if not specs:
                continue

            recovery_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a mandatory ML tool caller. "
                        "Call the only available tool exactly once. "
                        'Use exactly {"run": true}. '
                        "Do not answer in prose and do not provide ML features."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Call {tool_name} now. "
                        'Arguments must be exactly {"run": true}.'
                    ),
                },
            ]

            for _ in range(2):
                recovery_rounds += 1

                try:
                    reply = self.client.chat(
                        model=model,
                        messages=recovery_messages,
                        tools=specs,
                        temperature=self.s.agent_temperature,
                    )
                except OllamaError as exc:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Qwen recovery runtime error for {tool_name}: {exc}"
                        ),
                    })
                    break

                recovery_messages.append(reply)
                calls = reply.get("tool_calls") or []

                selected = None

                for call in calls:
                    fn = call.get("function", {})
                    if str(fn.get("name", "")) == tool_name:
                        selected = fn
                        break

                if selected is None:
                    recovery_messages.append({
                        "role": "user",
                        "content": (
                            f"Tool call masih belum ada. Call {tool_name} "
                            'exactly once with {"run": true}.'
                        ),
                    })
                    continue

                llm_args = self._arguments(
                    selected.get("arguments")
                )

                trusted_args = self._trusted_args(
                    tool_name,
                    feature_context,
                )

                trace = dispatch(
                    tool_name,
                    trusted_args,
                    execute=execute_tools,
                    caller="qwen",
                    llm_arguments=llm_args,
                    binding_source="python_feature_context",
                )

                record.add(trace)

                recovery_messages.append({
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": trace.model_content(),
                })

                messages.extend(recovery_messages)
                break

        return recovery_rounds

    def _python_fallback(
        self,
        record: ToolRecord,
        messages: list[dict[str, Any]],
        feature_context: dict[str, dict[str, Any]],
        *,
        execute_tools: bool,
    ) -> None:
        """Reliability fallback for ML tools only.

        RAG is intentionally NOT silently executed by Python because the user
        explicitly wants Qwen to be the agent that calls the RAG tool.
        """
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

    def _run_policy_rag(
        self,
        *,
        record: ToolRecord,
        messages: list[dict[str, Any]],
        policy_context: dict[str, Any],
        execute_tools: bool,
        model: str,
    ) -> int:
        """Run a separate Qwen tool-calling round for the policy RAG."""
        verified_ml = {
            name: record.last_success_by_name.get(name)
            for name in REQUIRED_ML_TOOLS
        }
        runtime_errors = {
            name: [
                t.error
                for t in record.traces
                if (
                    t.name == name
                    and t.error
                    and not t.duplicate_blocked
                )
            ]
            for name in REQUIRED_ML_TOOLS
        }

        policy_payload = {
            "task": "retrieve_applicable_credit_policy",
            "borrower_policy_context": policy_context or {},
            "verified_ml_results": verified_ml,
            "ml_runtime_errors": runtime_errors,
        }

        rag_messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": POLICY_SYSTEM,
            },
            {
                "role": "user",
                "content": json.dumps(
                    policy_payload,
                    ensure_ascii=False,
                    default=str,
                ),
            },
        ]

        max_rounds = max(
            1,
            int(getattr(self.s, "rag_tool_rounds", 2)),
        )

        for round_no in range(1, max_rounds + 1):
            try:
                reply = self.client.chat(
                    model=model,
                    messages=rag_messages,
                    tools=[rag_definition()],
                    temperature=self.s.agent_temperature,
                )
            except OllamaError as exc:
                trace = ToolTrace(
                    name=POLICY_RAG_TOOL,
                    arguments={},
                    error=f"Qwen RAG orchestration error: {exc}",
                    caller="qwen",
                    llm_arguments=None,
                    binding_source="qwen_policy_query",
                )
                record.add(trace)
                messages.extend(rag_messages)
                return round_no

            rag_messages.append(reply)
            calls = reply.get("tool_calls") or []

            valid_calls = []
            for call in calls:
                fn = call.get("function", {})
                name = str(fn.get("name", ""))
                args = self._arguments(
                    fn.get("arguments")
                )

                if name != POLICY_RAG_TOOL:
                    continue

                valid_calls.append((name, args))

            if valid_calls:
                name, args = valid_calls[0]

                # RAG is the one place Qwen IS allowed to generate arguments:
                # a natural-language policy query + optional top_k.
                trace = dispatch(
                    name,
                    args,
                    execute=execute_tools,
                    caller="qwen",
                    llm_arguments=args,
                    binding_source="qwen_policy_query",
                )
                record.add(trace)

                rag_messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "content": trace.model_content(),
                })

                messages.extend(rag_messages)
                return round_no

            rag_messages.append({
                "role": "user",
                "content": (
                    "You must call query_credit_policy exactly once. "
                    "Provide a non-empty Indonesian `query`."
                ),
            })

        trace = ToolTrace(
            name=POLICY_RAG_TOOL,
            arguments={},
            error=(
                "Qwen did not call query_credit_policy within "
                f"{max_rounds} policy round(s)."
            ),
            caller="qwen",
            llm_arguments=None,
            binding_source="qwen_policy_query",
        )
        record.add(trace)
        messages.extend(rag_messages)
        return max_rounds

    def run(
        self,
        feature_context: dict[str, dict[str, Any]],
        *,
        policy_context: dict[str, Any] | None = None,
        execute_tools: bool = True,
        model_override: str | None = None,
        use_rag: bool | None = None,
    ) -> AgentResult:
        model = model_override or self.s.qwen_agent_model

        payload = {
            "assessment_contract": {
                "required_ml_tools": list(REQUIRED_ML_TOOLS),
                "required_ml_tool_count": len(REQUIRED_ML_TOOLS),
                "ml_tool_arguments_policy": "run_true_only",
                "python_trusted_feature_binding": True,
            },
            "model_availability": self._agent_visible_summary(
                feature_context
            ),
        }

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": QWEN_SYSTEM,
            },
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

        # ====================================================
        # PHASE 1 — FOUR ML TOOLS
        # ====================================================

        for round_no in range(
            1,
            self.s.max_tool_rounds + 1,
        ):
            last_round = round_no

            try:
                reply = self.client.chat(
                    model=model,
                    messages=messages,
                    tools=ml_definitions(),
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

            not_attempted = sorted(
                set(REQUIRED_ML_TOOLS) - attempted
            )

            # All four were SELECTED by Qwen. Runtime failure does not require
            # Qwen to keep calling the same model over and over.
            if not not_attempted:
                break

            messages.append({
                "role": "user",
                "content": (
                    "ASSESSMENT INCOMPLETE. Call these remaining ML tools "
                    'with exactly {"run": true}: ' 
                    + ", ".join(not_attempted)
                ),
            })

        # Give Qwen a targeted real tool-calling recovery opportunity
        # before falling back to Python.
        recovery_rounds = self._qwen_recover_missing_ml_tools(
            record=record,
            messages=messages,
            feature_context=feature_context,
            execute_tools=execute_tools,
            model=model,
        )
        last_round += recovery_rounds

        # Last-resort reliability fallback only if Qwen still omitted a tool.
        self._python_fallback(
            record,
            messages,
            feature_context,
            execute_tools=execute_tools,
        )

        # ====================================================
        # PHASE 2 — POLICY RAG TOOL CALL BY QWEN
        # ====================================================

        rag_enabled = (
            bool(getattr(self.s, "rag_enabled", True))
            if use_rag is None
            else bool(use_rag)
        )

        # If caller supplies policy_context, the pipeline is asking for policy RAG.
        if rag_enabled and policy_context is not None:
            rag_rounds = self._run_policy_rag(
                record=record,
                messages=messages,
                policy_context=policy_context,
                execute_tools=execute_tools,
                model=model,
            )
            last_round += rag_rounds

        # ====================================================
        # FINAL STATUS
        # ====================================================

        ml_failed = sorted(
            set(REQUIRED_ML_TOOLS)
            - set(record.last_success_by_name)
        )

        rag_expected = (
            rag_enabled
            and policy_context is not None
        )

        rag_result = record.rag_result or {}
        rag_runtime_ok = (
            not rag_expected
            or (
                record.rag_qwen_attempted
                and bool(rag_result)
            )
        )
        rag_retrieved = (
            not rag_expected
            or rag_result.get("status")
            in {"retrieved", "no_match"}
        )

        if ml_failed and not rag_runtime_ok:
            stopped_reason = "model_and_policy_runtime_failure"
        elif ml_failed:
            stopped_reason = "model_runtime_failure"
        elif not rag_runtime_ok:
            stopped_reason = "policy_rag_runtime_failure"
        elif not rag_retrieved:
            stopped_reason = "policy_rag_not_ready"
        else:
            stopped_reason = "assessment_completed"

        return AgentResult(
            record=record,
            messages=messages,
            final_content=(
                "Credit-risk ML tool orchestration and policy RAG phase completed."
            ),
            rounds=last_round,
            stopped_reason=stopped_reason,
        )
