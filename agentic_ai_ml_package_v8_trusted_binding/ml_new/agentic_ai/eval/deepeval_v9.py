from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from ..config import SETTINGS
from ..document_extraction import (
    DocumentExtractionResult,
    ExtractedPage,
)
from ..document_mapper import DeterministicDocumentMapper
from ..schemas import (
    POLICY_RAG_TOOL,
    REQUIRED_AGENT_TOOLS,
    REQUIRED_ML_TOOLS,
)
from ..tool_registry import TOOL_TO_MODEL


def load_v7_goldens(
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    path = (
        Path(path)
        if path
        else Path(__file__).with_name("goldens_v7.jsonl")
    )
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def _same_value(
    actual: Any,
    expected: Any,
    *,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-6,
) -> bool:
    if isinstance(expected, bool):
        return actual is expected or bool(actual) is expected

    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        try:
            return math.isclose(
                float(actual),
                float(expected),
                rel_tol=rel_tol,
                abs_tol=max(
                    abs_tol,
                    abs(float(expected)) * 1e-9,
                ),
            )
        except Exception:
            return False

    return (
        str(actual).strip()
        == str(expected).strip()
    )


def evaluate_document_mapper(
    *,
    goldens_path: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    mapper = DeterministicDocumentMapper()
    cases = load_v7_goldens(goldens_path)

    total_expected = 0
    total_correct = 0
    total_returned_relevant = 0
    total_false_positive = 0
    forbidden_hits = 0
    case_rows: list[dict[str, Any]] = []

    for case in cases:
        docs = DocumentExtractionResult(
            pages=[
                ExtractedPage(
                    source_name=case["source_name"],
                    page=case.get("page"),
                    text=case["page_text"],
                    method="golden",
                )
            ],
            warnings=[],
        )

        extraction = mapper.extract(docs)
        actual = {
            k: v.value
            for k, v in extraction.raw_facts.items()
        }
        expected = case.get(
            "expected_raw_facts",
            {},
        )
        forbidden = set(
            case.get("forbidden_facts", [])
        )

        checks = {
            key: (
                key in actual
                and _same_value(
                    actual[key],
                    expected_value,
                )
            )
            for key, expected_value
            in expected.items()
        }

        correct = sum(checks.values())
        expected_n = len(expected)
        relevant_actual = {
            k for k in actual
            if k in expected or k in forbidden
        }
        false_positive = sum(
            1
            for k in relevant_actual
            if k not in expected
        )
        forbidden_case_hits = sum(
            1 for k in forbidden
            if k in actual
        )

        total_expected += expected_n
        total_correct += correct
        total_returned_relevant += (
            correct + false_positive
        )
        total_false_positive += false_positive
        forbidden_hits += forbidden_case_hits

        recall = (
            correct / expected_n
            if expected_n
            else 1.0
        )
        precision = (
            correct / (correct + false_positive)
            if (correct + false_positive)
            else 1.0
        )

        case_rows.append({
            "id": case["id"],
            "expected_fields": expected_n,
            "correct_fields": correct,
            "field_recall": recall,
            "field_precision": precision,
            "forbidden_hits": forbidden_case_hits,
            "checks": checks,
        })

        if verbose:
            print(
                f"{case['id']:<22} "
                f"recall={recall:.3f} "
                f"precision={precision:.3f} "
                f"forbidden_hits={forbidden_case_hits}"
            )

    recall = (
        total_correct / total_expected
        if total_expected
        else 1.0
    )
    precision = (
        total_correct / total_returned_relevant
        if total_returned_relevant
        else 1.0
    )
    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "layer": "document_mapper",
        "cases": len(cases),
        "field_recall": round(recall, 6),
        "field_precision": round(precision, 6),
        "field_f1": round(f1, 6),
        "correct_fields": total_correct,
        "expected_fields": total_expected,
        "false_positive_fields": total_false_positive,
        "forbidden_hallucination_hits": forbidden_hits,
        "case_results": case_rows,
    }


def _json_exact(a: Any, b: Any) -> bool:
    return json.dumps(
        a,
        sort_keys=True,
        default=str,
    ) == json.dumps(
        b,
        sort_keys=True,
        default=str,
    )


def _judge_model(
    judge_model: str | None = None,
):
    try:
        from deepeval.models import OllamaModel
    except ImportError as exc:
        raise RuntimeError(
            "DeepEval/Ollama dependency belum lengkap. "
            "Install: pip install deepeval==4.1.8 ollama"
        ) from exc

    model_name = (
        judge_model
        or os.getenv(
            "DEEPEVAL_JUDGE_MODEL",
            "",
        ).strip()
        or SETTINGS.qwen_agent_model
    )

    return OllamaModel(
        model=model_name,
        base_url=SETTINGS.ollama_host,
        temperature=0,
    )


def evaluate_qwen_tool_calling(
    *,
    agent_result: Any,
    feature_context: dict[str, dict[str, Any]],
    judge_model: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """V9 evaluation.

    - Qwen ML phase: correct 4 ML tool names with exact trigger argument {"run": true}.
    - Python: exact trusted feature binding for the 4 ML tools.
    - Qwen RAG phase: query_credit_policy called with non-empty query.
    """
    try:
        from deepeval.metrics import ToolCorrectnessMetric
        from deepeval.test_case import LLMTestCase, ToolCall
    except ImportError as exc:
        raise RuntimeError("deepeval belum terinstall.") from exc

    qwen_traces = [
        t
        for t in agent_result.record.traces
        if (
            t.name in REQUIRED_AGENT_TOOLS
            and t.caller == "qwen"
            and not t.duplicate_blocked
        )
    ]

    # Tool selection correctness: compare NAMES only.
    tools_called = [
        ToolCall(name=t.name, input_parameters={})
        for t in qwen_traces
    ]
    expected_tools = [
        ToolCall(name=name, input_parameters={})
        for name in REQUIRED_AGENT_TOOLS
    ]

    metric = ToolCorrectnessMetric(
        threshold=1.0,
        model=_judge_model(judge_model),
        should_exact_match=True,
        should_consider_ordering=False,
        include_reason=False,
        strict_mode=True,
    )

    test_case = LLMTestCase(
        input=(
            "Call the four mandatory ML tools, then call "
            "query_credit_policy once for policy retrieval."
        ),
        actual_output="Qwen multi-phase tool-selection trace.",
        tools_called=tools_called,
        expected_tools=expected_tools,
    )

    metric.measure(test_case)

    qwen_names = [t.name for t in qwen_traces]
    required_agent = set(REQUIRED_AGENT_TOOLS)
    required_ml = set(REQUIRED_ML_TOOLS)

    agent_coverage = (
        len(set(qwen_names) & required_agent)
        / len(required_agent)
    )
    ml_coverage = (
        len(set(qwen_names) & required_ml)
        / len(required_ml)
    )

    duplicates = max(0, len(qwen_names) - len(set(qwen_names)))
    invalid = sum(1 for name in qwen_names if name not in required_agent)

    # ML LLM argument policy
    ml_traces = [
        t for t in qwen_traces
        if t.name in REQUIRED_ML_TOOLS
    ]
    ml_arg_policy = {
        t.name: (
            set((t.llm_arguments or {}).keys()) == {"run"}
            and (t.llm_arguments or {}).get("run") is True
        )
        for t in ml_traces
    }
    ml_arg_compliance = (
        sum(ml_arg_policy.values())
        / len(REQUIRED_ML_TOOLS)
    )

    # Trusted Python feature binding fidelity
    binding_match: dict[str, bool] = {}
    for tool_name in REQUIRED_ML_TOOLS:
        traces = [
            t for t in ml_traces
            if t.name == tool_name
        ]
        if not traces:
            binding_match[tool_name] = False
            continue

        model_key = TOOL_TO_MODEL[tool_name]
        expected_bound = {
            "features": (
                feature_context
                .get(model_key, {})
                .get("features", {})
            )
        }

        binding_match[tool_name] = (
            traces[0].binding_source == "python_feature_context"
            and _json_exact(
                traces[0].arguments,
                expected_bound,
            )
        )

    binding_fidelity = (
        sum(binding_match.values())
        / len(REQUIRED_ML_TOOLS)
    )

    # RAG query diagnostics
    rag_traces = [
        t for t in qwen_traces
        if t.name == POLICY_RAG_TOOL
    ]
    rag_attempted = bool(rag_traces)
    rag_query = ""
    rag_query_nonempty = False
    rag_status = None
    rag_citation_count = 0

    if rag_traces:
        rag_query = str(
            (rag_traces[0].arguments or {}).get("query") or ""
        ).strip()
        rag_query_nonempty = bool(rag_query)
        rag_result = rag_traces[0].result or {}
        rag_status = rag_result.get("status")
        rag_citation_count = int(
            rag_result.get("citation_count")
            or len(rag_result.get("citations") or [])
        )

    result = {
        "layer": "qwen_tool_calling_plus_rag",
        "deepeval_tool_correctness": float(metric.score),
        "qwen_agent_tool_coverage": round(agent_coverage, 6),
        "qwen_ml_tool_coverage": round(ml_coverage, 6),
        "duplicate_count": duplicates,
        "invalid_tool_count": invalid,
        "qwen_ml_run_trigger_compliance": round(
            ml_arg_compliance,
            6,
        ),
        "python_binding_fidelity": round(
            binding_fidelity,
            6,
        ),
        "python_binding_match_by_tool": binding_match,
        "rag_attempted_by_qwen": rag_attempted,
        "rag_query_nonempty": rag_query_nonempty,
        "rag_query": rag_query,
        "rag_status": rag_status,
        "rag_citation_count": rag_citation_count,
        "qwen_attempted_tools": qwen_names,
    }

    if verbose:
        print(
            "DeepEval ToolCorrectness      :",
            result["deepeval_tool_correctness"],
        )
        print(
            "Qwen agent tool coverage      :",
            result["qwen_agent_tool_coverage"],
        )
        print(
            "Qwen ML tool coverage         :",
            result["qwen_ml_tool_coverage"],
        )
        print(
            "Qwen ML run-trigger compliance:",
            result["qwen_ml_run_trigger_compliance"],
        )
        print(
            "Python binding fidelity       :",
            result["python_binding_fidelity"],
        )
        print(
            "RAG attempted by Qwen          :",
            result["rag_attempted_by_qwen"],
        )
        print(
            "RAG query non-empty            :",
            result["rag_query_nonempty"],
        )
        print(
            "RAG status                     :",
            result["rag_status"],
        )
        print(
            "RAG citations                  :",
            result["rag_citation_count"],
        )

    return result


def narrator_ground_truth(
    *,
    agent_result: Any,
    feature_context: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    verified = (
        agent_result.record.last_success_by_name
    )

    errors_by_tool: dict[str, list[str]] = {}

    for tool_name in REQUIRED_ML_TOOLS:
        errors_by_tool[tool_name] = [
            str(t.error)
            for t in agent_result.record.traces
            if (
                t.name == tool_name
                and t.error
                and not t.duplicate_blocked
            )
        ]

    return {
        "verified_tool_results": {
            name: verified.get(name)
            for name in REQUIRED_ML_TOOLS
        },
        "runtime_status": {
            name: (
                "success"
                if name in verified
                else "failed"
            )
            for name in REQUIRED_ML_TOOLS
        },
        "tool_errors": errors_by_tool,
        "policy_rag_result": agent_result.record.rag_result,
        "feature_completeness": {
            model_key: {
                "observed_feature_count":
                    model_ctx.get(
                        "observed_feature_count"
                    ),
                "expected_feature_count":
                    model_ctx.get(
                        "expected_feature_count"
                    ),
                "feature_completeness_percent":
                    model_ctx.get(
                        "feature_completeness_percent"
                    ),
            }
            for model_key, model_ctx
            in feature_context.items()
        },
    }



def evaluate_feature_completeness(
    *,
    feature_context: dict[str, dict[str, Any]],
    verbose: bool = True,
) -> dict[str, Any]:
    """Deterministic input-completeness reporting.

    This metric reports observed/expected feature coverage only. It does NOT
    infer prediction accuracy or model reliability from completeness.
    """
    models: dict[str, dict[str, Any]] = {}

    for model_key, ctx in feature_context.items():
        observed = ctx.get("observed_feature_count")
        expected = ctx.get("expected_feature_count")
        pct = ctx.get("feature_completeness_percent")

        models[model_key] = {
            "observed_feature_count": observed,
            "expected_feature_count": expected,
            "feature_completeness_percent": pct,
        }

    result = {
        "layer": "input_feature_completeness",
        "models": models,
        "interpretation_rule": (
            "report_only_do_not_infer_accuracy"
        ),
    }

    if verbose:
        print("Input feature completeness:")
        for model_key, row in models.items():
            print(
                f" - {model_key:<12}: "
                f"{row['observed_feature_count']}/"
                f"{row['expected_feature_count']} "
                f"({row['feature_completeness_percent']}%)"
            )

    return result


def evaluate_narrator_groundedness(
    *,
    answer: str,
    agent_result: Any,
    feature_context: dict[str, dict[str, Any]],
    judge_model: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    try:
        from deepeval.metrics import GEval
        from deepeval.test_case import (
            LLMTestCase,
            SingleTurnParams,
        )
    except ImportError as exc:
        raise RuntimeError(
            "deepeval belum terinstall."
        ) from exc

    truth = narrator_ground_truth(
        agent_result=agent_result,
        feature_context=feature_context,
    )

    expected_output = json.dumps(
        truth,
        ensure_ascii=False,
        default=str,
        indent=2,
    )

    metric = GEval(
        name="CreditRiskNarratorFactualGrounding",
        criteria=(
            "Evaluate ONLY factual claim-to-evidence consistency. "
            "ACTUAL_OUTPUT is an Indonesian narrative and is NOT required to copy "
            "the JSON structure or mention every field in EXPECTED_OUTPUT. "
            "Do not penalize formatting differences or reasonable omissions. "

            "Feature completeness is input-quality metadata. Low completeness by itself "
            "is NOT a narrator-grounding failure and MUST NOT automatically lower the score. "
            "Do not infer that low completeness necessarily means predictions are inaccurate, "
            "unreliable, or unsuitable for financial decisions unless EXPECTED_OUTPUT explicitly "
            "contains that claim. "

            "Penalize ONLY unsupported factual claims, including invented prediction values, "
            "probabilities, thresholds, risk bands, clusters, model statuses, feature values, "
            "regulation names, article numbers, pages, policy thresholds, or citations. "
            "Penalize describing a failed model as successful, presenting missing/imputed values "
            "as directly observed, unsupported causal explanations, or treating an "
            "out-of-reference-range warning as proof that credit risk is high or low. "

            "A statement that an input is outside the model reference range is grounded if that "
            "warning exists in EXPECTED_OUTPUT; it is a data-quality warning, not automatically "
            "a risk conclusion. "

            "If some models succeeded and one failed, a partial assessment is valid. "
            "Policy statements must be grounded only in policy_rag_result. "
            "Score primarily on whether claims actually made by ACTUAL_OUTPUT are supported."
        ),
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.8,
        model=_judge_model(judge_model),
    )

    case = LLMTestCase(
        input=(
            "Produce a concise, grounded Indonesian "
            "credit-risk assessment from verified ML results."
        ),
        actual_output=str(answer or ""),
        expected_output=expected_output,
    )

    metric.measure(case)

    result = {
        "layer": "qwen_narrator",
        "deepeval_groundedness": float(metric.score),
        "deepeval_reason": getattr(
            metric,
            "reason",
            None,
        ),
        "pass_threshold": 0.8,
    }

    if verbose:
        print(
            "Narrator groundedness :",
            result["deepeval_groundedness"],
        )
        print(
            "Pass threshold        :",
            result["pass_threshold"],
        )
        if result["deepeval_reason"]:
            print(
                "Reason                :",
                result["deepeval_reason"],
            )

    return result


def evaluate_end_to_end(
    *,
    extraction: Any,
    feature_context: dict[str, dict[str, Any]],
    agent_result: Any,
    answer: str,
    mapper_eval: dict[str, Any] | None = None,
    tool_eval: dict[str, Any] | None = None,
    narrator_eval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    verified = (
        agent_result.record.last_success_by_name
    )

    successful_models = [
        name
        for name in REQUIRED_ML_TOOLS
        if name in verified
    ]
    failed_models = [
        name
        for name in REQUIRED_ML_TOOLS
        if name not in verified
    ]

    model_success_rate = (
        len(successful_models)
        / len(REQUIRED_ML_TOOLS)
    )

    completeness = {
        model_key:
            model_ctx.get(
                "feature_completeness_percent"
            )
        for model_key, model_ctx
        in feature_context.items()
    }

    return {
        "layer": "end_to_end",
        "borrower_name":
            getattr(
                extraction,
                "borrower_name",
                None,
            ),
        "raw_fact_count":
            len(
                getattr(
                    extraction,
                    "raw_facts",
                    {},
                )
                or {}
            ),
        "qwen_tool_coverage":
            round(
                float(
                    agent_result.record.qwen_coverage
                ),
                6,
            ),
        "successful_model_count":
            len(successful_models),
        "required_model_count":
            len(REQUIRED_ML_TOOLS),
        "model_success_rate":
            round(
                model_success_rate,
                6,
            ),
        "successful_models":
            successful_models,
        "failed_models":
            failed_models,
        "feature_completeness_percent":
            completeness,
        "final_answer_nonempty":
            bool(str(answer or "").strip()),
        "agent_stopped_reason":
            getattr(
                agent_result,
                "stopped_reason",
                None,
            ),
        "mapper_eval": mapper_eval,
        "tool_eval": tool_eval,
        "narrator_eval": narrator_eval,
    }


def save_evaluation_report(
    report: dict[str, Any],
    path: str | Path = (
        "agentic_ai_v8_evaluation_report.json"
    ),
) -> Path:
    path = Path(path)
    path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return path
