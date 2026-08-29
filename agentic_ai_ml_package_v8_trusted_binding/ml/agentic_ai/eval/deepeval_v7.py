from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from ..config import SETTINGS
from ..document_extraction import DocumentExtractionResult, ExtractedPage
from ..document_mapper import DeterministicDocumentMapper
from ..schemas import REQUIRED_ML_TOOLS


TOOL_TO_MODEL = {
    "predict_pd": "pd",
    "predict_ews": "ews",
    "predict_lgd": "lgd",
    "predict_pd_cluster": "pd_cluster",
}


def load_v7_goldens(path: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(path) if path else Path(__file__).with_name("goldens_v7.jsonl")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _same_value(actual: Any, expected: Any, *, rel_tol: float = 1e-6, abs_tol: float = 1e-6) -> bool:
    if isinstance(expected, bool):
        return actual is expected or bool(actual) is expected

    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return math.isclose(
                float(actual),
                float(expected),
                rel_tol=rel_tol,
                abs_tol=max(abs_tol, abs(float(expected)) * 1e-9),
            )
        except Exception:
            return False

    return str(actual).strip() == str(expected).strip()


def evaluate_document_mapper(
    *,
    goldens_path: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Layer 1: deterministic document-field extraction evaluation.

    This is intentionally deterministic rather than LLM-as-a-judge:
    exact field/numeric extraction should be measured against golden truth.
    """
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
        actual = {k: v.value for k, v in extraction.raw_facts.items()}
        expected = case.get("expected_raw_facts", {})
        forbidden = set(case.get("forbidden_facts", []))

        checks = {
            key: key in actual and _same_value(actual[key], expected_value)
            for key, expected_value in expected.items()
        }

        correct = sum(checks.values())
        expected_n = len(expected)

        # Precision is scoped to golden-known fields so bookkeeping/company-name
        # fields do not unfairly count as hallucinations.
        relevant_actual = {k for k in actual if k in expected or k in forbidden}
        false_positive = sum(1 for k in relevant_actual if k not in expected)
        forbidden_case_hits = sum(1 for k in forbidden if k in actual)

        total_expected += expected_n
        total_correct += correct
        total_returned_relevant += correct + false_positive
        total_false_positive += false_positive
        forbidden_hits += forbidden_case_hits

        recall = correct / expected_n if expected_n else 1.0
        precision = (
            correct / (correct + false_positive)
            if (correct + false_positive)
            else 1.0
        )

        row = {
            "id": case["id"],
            "expected_fields": expected_n,
            "correct_fields": correct,
            "field_recall": recall,
            "field_precision": precision,
            "forbidden_hits": forbidden_case_hits,
            "checks": checks,
        }
        case_rows.append(row)

        if verbose:
            print(
                f"{case['id']:<22} "
                f"recall={recall:.3f} precision={precision:.3f} "
                f"forbidden_hits={forbidden_case_hits}"
            )
            if expected:
                print("  checks:", checks)

    recall = total_correct / total_expected if total_expected else 1.0
    precision = (
        total_correct / total_returned_relevant
        if total_returned_relevant
        else 1.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
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


def _expected_tool_args(feature_context: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        tool_name: {
            "features": feature_context[model_key].get("features", {})
        }
        for tool_name, model_key in TOOL_TO_MODEL.items()
    }


def _dict_exact(a: Any, b: Any) -> bool:
    try:
        return json.dumps(a, sort_keys=True, default=str) == json.dumps(
            b, sort_keys=True, default=str
        )
    except Exception:
        return a == b


def evaluate_qwen_tool_calling(
    *,
    agent_result: Any,
    feature_context: dict[str, dict[str, Any]],
    verbose: bool = True,
) -> dict[str, Any]:
    """Layer 2: DeepEval ToolCorrectness + deterministic routing diagnostics.

    Only Qwen-originated calls are evaluated. Python fallback calls are excluded,
    because this layer is specifically testing the LLM agent.
    """
    try:
        from deepeval.metrics import ToolCorrectnessMetric
        from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams
    except ImportError as exc:
        raise RuntimeError(
            "deepeval belum terinstall. Jalankan: pip install deepeval==4.1.8"
        ) from exc

    qwen_traces = [
        t for t in agent_result.record.traces
        if (
            t.name in REQUIRED_ML_TOOLS
            and t.caller == "qwen"
            and not t.duplicate_blocked
        )
    ]

    expected_args = _expected_tool_args(feature_context)

    tools_called = [
        ToolCall(name=t.name, input_parameters=t.arguments)
        for t in qwen_traces
    ]
    expected_tools = [
        ToolCall(name=name, input_parameters=expected_args[name])
        for name in REQUIRED_ML_TOOLS
    ]

    metric = ToolCorrectnessMetric(
        threshold=1.0,
        should_exact_match=True,
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
    )

    test_case = LLMTestCase(
        input=(
            "Run the mandatory holistic credit risk assessment by calling exactly "
            "predict_pd, predict_ews, predict_lgd, and predict_pd_cluster. "
            "Copy only the available model features supplied by Python."
        ),
        actual_output="Qwen tool-calling trace",
        tools_called=tools_called,
        expected_tools=expected_tools,
    )

    metric.measure(test_case)

    qwen_names = [t.name for t in qwen_traces]
    qwen_unique = set(qwen_names)
    required = set(REQUIRED_ML_TOOLS)

    coverage = len(qwen_unique & required) / len(required)
    exact_tool_set = float(len(qwen_names) == len(required) and qwen_unique == required)
    duplicate_count = max(0, len(qwen_names) - len(qwen_unique))
    invalid_tool_count = sum(1 for name in qwen_names if name not in required)

    per_tool_argument_match: dict[str, bool] = {}
    for tool_name in REQUIRED_ML_TOOLS:
        traces = [t for t in qwen_traces if t.name == tool_name]
        if not traces:
            per_tool_argument_match[tool_name] = False
            continue
        # The first non-duplicate Qwen call is the agent's intended call.
        per_tool_argument_match[tool_name] = _dict_exact(
            traces[0].arguments,
            expected_args[tool_name],
        )

    arg_fidelity = (
        sum(per_tool_argument_match.values()) / len(REQUIRED_ML_TOOLS)
    )

    result = {
        "layer": "qwen_tool_calling",
        "deepeval_tool_correctness": float(metric.score),
        "deepeval_reason": getattr(metric, "reason", None),
        "qwen_tool_coverage": round(coverage, 6),
        "exact_tool_set": exact_tool_set,
        "argument_fidelity": round(arg_fidelity, 6),
        "per_tool_argument_match": per_tool_argument_match,
        "duplicate_count": duplicate_count,
        "invalid_tool_count": invalid_tool_count,
        "qwen_attempted_tools": qwen_names,
    }

    if verbose:
        print("DeepEval ToolCorrectness :", result["deepeval_tool_correctness"])
        print("Qwen tool coverage       :", result["qwen_tool_coverage"])
        print("Exact mandatory tool set :", result["exact_tool_set"])
        print("Argument fidelity        :", result["argument_fidelity"])
        print("Duplicate calls          :", result["duplicate_count"])
        print("Invalid calls            :", result["invalid_tool_count"])
        print("Per-tool argument match  :", per_tool_argument_match)

    return result


def _judge_model(judge_model: str | None = None):
    try:
        from deepeval.models import OllamaModel
    except ImportError as exc:
        raise RuntimeError(
            "deepeval belum terinstall. Jalankan: pip install deepeval==4.1.8"
        ) from exc

    model_name = (
        judge_model
        or os.getenv("DEEPEVAL_JUDGE_MODEL", "").strip()
        or SETTINGS.qwen_agent_model
    )

    return OllamaModel(
        model=model_name,
        base_url=SETTINGS.ollama_host,
        temperature=0,
    )


def narrator_ground_truth(
    *,
    agent_result: Any,
    feature_context: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    verified = agent_result.record.last_success_by_name

    errors_by_tool: dict[str, list[str]] = {}
    for tool_name in REQUIRED_ML_TOOLS:
        errors_by_tool[tool_name] = [
            str(t.error)
            for t in agent_result.record.traces
            if t.name == tool_name and t.error and not t.duplicate_blocked
        ]

    return {
        "verified_tool_results": {
            name: verified.get(name)
            for name in REQUIRED_ML_TOOLS
        },
        "tool_errors": errors_by_tool,
        "feature_completeness": {
            model_key: {
                "observed_feature_count": model_ctx.get("observed_feature_count"),
                "expected_feature_count": model_ctx.get("expected_feature_count"),
                "feature_completeness_percent": model_ctx.get(
                    "feature_completeness_percent"
                ),
                "missing_feature_names": model_ctx.get(
                    "missing_feature_names", []
                ),
            }
            for model_key, model_ctx in feature_context.items()
        },
    }


def evaluate_narrator_groundedness(
    *,
    answer: str,
    agent_result: Any,
    feature_context: dict[str, dict[str, Any]],
    judge_model: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Layer 3: DeepEval GEval for the final SahabatAI risk narration."""
    try:
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, SingleTurnParams
    except ImportError as exc:
        raise RuntimeError(
            "deepeval belum terinstall. Jalankan: pip install deepeval==4.1.8"
        ) from exc

    ground_truth = narrator_ground_truth(
        agent_result=agent_result,
        feature_context=feature_context,
    )

    expected_output = json.dumps(
        ground_truth,
        ensure_ascii=False,
        default=str,
        indent=2,
    )

    judge = _judge_model(judge_model)

    faithfulness = GEval(
        name="CreditRiskNarratorGroundedness",
        criteria=(
            "Evaluate whether ACTUAL_OUTPUT is fully grounded in EXPECTED_OUTPUT. "
            "Penalize any invented probability, LGD value, risk band, cluster, "
            "feature value, model success, or causal explanation that is not supported. "
            "If a model has no verified result, the narration must not present that model "
            "as successfully scored. Missing/imputed inputs must not be described as "
            "directly observed facts."
        ),
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.8,
        model=judge,
    )

    test_case = LLMTestCase(
        input="Produce the final grounded Indonesian holistic credit risk assessment.",
        actual_output=answer,
        expected_output=expected_output,
    )

    faithfulness.measure(test_case)

    result = {
        "layer": "sahabat_narrator",
        "deepeval_groundedness": float(faithfulness.score),
        "deepeval_reason": getattr(faithfulness, "reason", None),
        "judge_model": (
            judge_model
            or os.getenv("DEEPEVAL_JUDGE_MODEL", "").strip()
            or SETTINGS.qwen_agent_model
        ),
        "pass_threshold": 0.8,
    }

    if verbose:
        print("Narrator groundedness :", result["deepeval_groundedness"])
        print("Pass threshold        :", result["pass_threshold"])
        print("Judge model           :", result["judge_model"])
        if result["deepeval_reason"]:
            print("Reason                :", result["deepeval_reason"])

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
    """Layer 4: non-arbitrary end-to-end scorecard.

    We intentionally do NOT collapse heterogeneous metrics into one weighted score.
    The report exposes each gate separately.
    """
    verified = agent_result.record.last_success_by_name
    successful_models = [name for name in REQUIRED_ML_TOOLS if name in verified]
    failed_models = [name for name in REQUIRED_ML_TOOLS if name not in verified]

    model_success_rate = len(successful_models) / len(REQUIRED_ML_TOOLS)
    qwen_coverage = float(agent_result.record.qwen_coverage)

    completeness = {
        model_key: model_ctx.get("feature_completeness_percent")
        for model_key, model_ctx in feature_context.items()
    }

    return {
        "layer": "end_to_end",
        "borrower_name": getattr(extraction, "borrower_name", None),
        "raw_fact_count": len(getattr(extraction, "raw_facts", {}) or {}),
        "qwen_tool_coverage": round(qwen_coverage, 6),
        "successful_model_count": len(successful_models),
        "required_model_count": len(REQUIRED_ML_TOOLS),
        "model_success_rate": round(model_success_rate, 6),
        "successful_models": successful_models,
        "failed_models": failed_models,
        "feature_completeness_percent": completeness,
        "final_answer_nonempty": bool(str(answer or "").strip()),
        "agent_stopped_reason": getattr(agent_result, "stopped_reason", None),
        "mapper_eval": mapper_eval,
        "tool_eval": tool_eval,
        "narrator_eval": narrator_eval,
    }


def save_evaluation_report(
    report: dict[str, Any],
    path: str | Path = "agentic_ai_v7_evaluation_report.json",
) -> Path:
    path = Path(path)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path
