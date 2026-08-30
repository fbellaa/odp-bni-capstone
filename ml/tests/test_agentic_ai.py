from __future__ import annotations

from types import SimpleNamespace

from ml.agentic_ai.feature_engineering import FeatureEngineer
from ml.agentic_ai.schemas import BorrowerExtraction, ExtractedValue, REQUIRED_ML_TOOLS
from ml.agentic_ai.tool_registry import TOOL_TO_MODEL


class FakeStore:
    def feature_defs(self, key):
        # Minimal fake schema to test deterministic feature building without real artifacts.
        if key == "pd":
            return (
                SimpleNamespace(name="der", json_type=lambda: "number"),
                SimpleNamespace(name="sales", json_type=lambda: "number"),
            )
        return (SimpleNamespace(name=f"{key}_raw", json_type=lambda: "number"),)


def test_required_tool_contract():
    assert tuple(REQUIRED_ML_TOOLS) == (
        "predict_pd",
        "predict_ews",
        "predict_lgd",
        "predict_pd_cluster",
    )
    assert set(TOOL_TO_MODEL) == set(REQUIRED_ML_TOOLS)


def test_deterministic_feature_engineering_der():
    extraction = BorrowerExtraction(
        raw_facts={
            "interest_bearing_debt": ExtractedValue(value=60.0),
            "equity": ExtractedValue(value=30.0),
            "sales": ExtractedValue(value=100.0),
        }
    )
    report = FeatureEngineer(store=FakeStore()).build(extraction)
    assert report["pd"]["features"]["der"] == 2.0
    assert report["pd"]["features"]["sales"] == 100.0
    assert report["pd"]["feature_provenance"]["der"]["source"] == "deterministic_feature_engineering"


def test_missing_features_are_preserved_not_invented():
    extraction = BorrowerExtraction(raw_facts={})
    report = FeatureEngineer(store=FakeStore()).build(extraction)
    assert "der" in report["pd"]["missing_feature_names"]
    assert "sales" in report["pd"]["missing_feature_names"]


def test_prefixed_financial_alias_and_ratio():
    class Store:
        def feature_defs(self, key):
            if key == "pd":
                return (
                    SimpleNamespace(name="fin_total_aset_rp", json_type=lambda: "number"),
                    SimpleNamespace(name="fin_current_ratio", json_type=lambda: "number"),
                )
            return ()

    extraction = BorrowerExtraction(
        raw_facts={
            "total_assets": ExtractedValue(value=366_000_000_000),
            "current_assets": ExtractedValue(value=202_000_000_000),
            "current_liabilities": ExtractedValue(value=128_000_000_000),
        }
    )
    report = FeatureEngineer(store=Store()).build(extraction)
    assert report["pd"]["features"]["fin_total_aset_rp"] == 366_000_000_000.0
    assert report["pd"]["features"]["fin_current_ratio"] == 202 / 128
    assert report["pd"]["model_can_attempt_with_imputation"] is True


def test_document_reducer_keeps_financial_lines():
    from ml.agentic_ai.document_extraction import DocumentExtractionResult, ExtractedPage
    from ml.agentic_ai.document_reducer import reduce_documents_for_extraction

    docs = DocumentExtractionResult(
        pages=[
            ExtractedPage(
                source_name="lk.pdf",
                page=1,
                method="pypdf",
                text=(
                    "PT SAGARA PRIMA\n"
                    "NERACA DALAM RP JUTA\n"
                    "Baris tidak relevan\n" * 40
                    + "TOTAL AKTIVA 366.000\n"
                    + "TOTAL HUTANG LANCAR 128.000\n"
                    + "LABA TAHUN BERJALAN 24.000\n"
                ),
            )
        ]
    )
    reduced = reduce_documents_for_extraction(docs, max_chars=5000)
    assert "TOTAL AKTIVA 366.000" in reduced.text
    assert "LABA TAHUN BERJALAN 24.000" in reduced.text
    assert reduced.reduced_chars <= reduced.original_chars


def test_v7_document_mapper_sagara_financials():
    from ml.agentic_ai.document_extraction import DocumentExtractionResult, ExtractedPage
    from ml.agentic_ai.document_mapper import DeterministicDocumentMapper

    docs = DocumentExtractionResult(
        pages=[
            ExtractedPage(
                source_name="lk.pdf",
                page=1,
                method="pypdf",
                text=(
                    "PT SAGARA PRIMA INFRASTRUKTUR\n"
                    "NERACA PER 31 MEI 2026\n"
                    "Dalam Rp juta\n"
                    "Total Aktiva Lancar 202.000\n"
                    "TOTAL AKTIVA 366.000\n"
                    "Total Hutang Lancar 128.000\n"
                    "Hutang Bank 38.000\n"
                    "Hutang Bank Jangka Panjang 64.000\n"
                    "Hutang Pembiayaan Konsumen 9.000\n"
                    "Total Hutang Jangka Panjang 78.000\n"
                    "Laba Ditahan 33.000\n"
                    "Total Modal dan Laba 160.000\n"
                ),
            ),
            ExtractedPage(
                source_name="lk.pdf",
                page=2,
                method="pypdf",
                text=(
                    "PT SAGARA PRIMA INFRASTRUKTUR\n"
                    "LAPORAN LABA / RUGI - Dalam Rp juta\n"
                    "Hasil Termijn Bersih 218.000\n"
                    "LABA BRUTO 104.000\n"
                    "Total Biaya Proyek Tidak Langsung 21.000\n"
                    "Total Biaya Administrasi dan Umum 33.000\n"
                    "Biaya Bunga Bank 13.500\n"
                    "LABA TAHUN BERJALAN 24.000\n"
                ),
            ),
            ExtractedPage(
                source_name="lk.pdf",
                page=3,
                method="pypdf",
                text=(
                    "Dalam Rp juta\n"
                    "Total Setara Kas 18.500\n"
                    "Total Piutang Proyek 100.000\n"
                    "Total Persediaan 31.000\n"
                ),
            ),
        ]
    )
    x = DeterministicDocumentMapper().extract(docs)
    assert x.borrower_name == "PT SAGARA PRIMA INFRASTRUKTUR"
    assert x.raw_facts["total_assets"].value == 366_000_000_000
    assert x.raw_facts["current_assets"].value == 202_000_000_000
    assert x.raw_facts["current_liabilities"].value == 128_000_000_000
    assert x.raw_facts["total_liabilities"].value == 206_000_000_000
    assert x.raw_facts["equity"].value == 160_000_000_000
    assert x.raw_facts["revenue"].value == 218_000_000_000
    assert x.raw_facts["gross_profit"].value == 104_000_000_000
    assert x.raw_facts["net_income"].value == 24_000_000_000
    assert x.raw_facts["interest_bearing_debt"].value == 111_000_000_000
    assert x.raw_facts["operating_profit"].value == 50_000_000_000


def test_v7_financial_rules_from_mapper_feed_model_features():
    from ml.agentic_ai.document_extraction import DocumentExtractionResult, ExtractedPage
    from ml.agentic_ai.document_mapper import DeterministicDocumentMapper

    class Store:
        def feature_defs(self, key):
            if key != "pd":
                return ()
            return (
                SimpleNamespace(name="fin_total_aset_rp", json_type=lambda: "number"),
                SimpleNamespace(name="fin_penjualan_rp", json_type=lambda: "number"),
                SimpleNamespace(name="fin_asset_turnover", json_type=lambda: "number"),
                SimpleNamespace(name="fin_current_ratio", json_type=lambda: "number"),
                SimpleNamespace(name="fin_quick_ratio", json_type=lambda: "number"),
                SimpleNamespace(name="fin_gross_margin", json_type=lambda: "number"),
                SimpleNamespace(name="fin_operating_margin", json_type=lambda: "number"),
                SimpleNamespace(name="fin_roa", json_type=lambda: "number"),
                SimpleNamespace(name="fin_re_to_ta", json_type=lambda: "number"),
                SimpleNamespace(name="fin_wc_to_ta", json_type=lambda: "number"),
                SimpleNamespace(name="fin_der", json_type=lambda: "number"),
            )

    docs = DocumentExtractionResult(pages=[
        ExtractedPage(source_name="x.pdf", page=1, method="pypdf", text=(
            "Dalam Rp juta\n"
            "Total Aktiva Lancar 202.000\nTOTAL AKTIVA 366.000\nTotal Hutang Lancar 128.000\n"
            "Hutang Bank 38.000\nHutang Bank Jangka Panjang 64.000\nHutang Pembiayaan Konsumen 9.000\n"
            "Total Hutang Jangka Panjang 78.000\nLaba Ditahan 33.000\nTotal Modal dan Laba 160.000\n"
            "Total Persediaan 31.000\nHasil Termijn Bersih 218.000\nLABA BRUTO 104.000\n"
            "Total Biaya Proyek Tidak Langsung 21.000\nTotal Biaya Administrasi dan Umum 33.000\n"
            "LABA TAHUN BERJALAN 24.000\n"
        ))
    ])
    extraction = DeterministicDocumentMapper().extract(docs)
    report = FeatureEngineer(store=Store()).build(extraction)
    feats = report["pd"]["features"]
    assert feats["fin_total_aset_rp"] == 366_000_000_000.0
    assert feats["fin_penjualan_rp"] == 218_000_000_000.0
    assert round(feats["fin_current_ratio"], 6) == round(202/128, 6)
    assert round(feats["fin_quick_ratio"], 6) == round((202-31)/128, 6)
    assert round(feats["fin_asset_turnover"], 6) == round(218/366, 6)
    assert round(feats["fin_gross_margin"], 6) == round(104/218, 6)
    assert round(feats["fin_operating_margin"], 6) == round(50/218, 6)
    assert round(feats["fin_roa"], 6) == round(24/366, 6)
    assert round(feats["fin_re_to_ta"], 6) == round(33/366, 6)
    assert round(feats["fin_wc_to_ta"], 6) == round((202-128)/366, 6)
    assert round(feats["fin_der"], 6) == round(111/160, 6)


def test_v9_2_tool_schema_uses_run_trigger_only():
    from ml.agentic_ai.tool_registry import definition

    spec = definition("predict_pd")
    params = spec["function"]["parameters"]

    assert set(params["properties"]) == {"run"}
    assert params["required"] == ["run"]
    assert params["additionalProperties"] is False


def test_v8_qwen_hallucinated_arguments_are_ignored(monkeypatch):
    from types import SimpleNamespace
    import ml.agentic_ai.agent as agent_module

    feature_context = {
        "pd": {"features": {"fin_current_ratio": 1.578125}, "missing_feature_names": []},
        "ews": {"features": {"perilaku_dpd": 12}, "missing_feature_names": []},
        "lgd": {"features": {"app_tenor_bulan": 12}, "missing_feature_names": []},
        "pd_cluster": {"features": {"app_skor_kredit": 742.0}, "missing_feature_names": []},
    }

    class FakeClient:
        def chat(self, **kwargs):
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "predict_pd",
                            "arguments": {"run": True, "features": {"fin_current_ratio": 999}},
                        }
                    },
                    {
                        "function": {
                            "name": "predict_ews",
                            "arguments": {"run": True, "features": {"wrong": 999}},
                        }
                    },
                    {
                        "function": {
                            "name": "predict_lgd",
                            "arguments": {"run": True, "features": {"wrong": 999}},
                        }
                    },
                    {
                        "function": {
                            "name": "predict_pd_cluster",
                            "arguments": {"run": True, "features": {"app_skor_kredit": 999}},
                        }
                    },
                ],
            }

    captured = []

    def fake_dispatch(
        name,
        arguments,
        *,
        execute=True,
        caller="unknown",
        llm_arguments=None,
        binding_source=None,
    ):
        from ml.agentic_ai.tool_registry import ToolTrace

        captured.append((name, arguments, llm_arguments, binding_source))
        return ToolTrace(
            name=name,
            arguments=arguments,
            result={"status": "ok"},
            caller=caller,
            llm_arguments=llm_arguments,
            binding_source=binding_source,
        )

    monkeypatch.setattr(agent_module, "dispatch", fake_dispatch)

    settings = SimpleNamespace(
        qwen_agent_model="fake",
        max_tool_rounds=1,
        agent_temperature=0,
    )

    result = agent_module.QwenMLAgent(
        client=FakeClient(),
        settings=settings,
    ).run(feature_context)

    by_name = {x[0]: x for x in captured}
    assert by_name["predict_pd"][1] == {
        "features": {"fin_current_ratio": 1.578125}
    }
    assert by_name["predict_ews"][1] == {
        "features": {"perilaku_dpd": 12}
    }
    assert by_name["predict_lgd"][1] == {
        "features": {"app_tenor_bulan": 12}
    }
    assert by_name["predict_pd_cluster"][1] == {
        "features": {"app_skor_kredit": 742.0}
    }

    # Raw hallucinated LLM payload is kept only for audit.
    assert by_name["predict_pd"][2]["features"]["fin_current_ratio"] == 999
    assert by_name["predict_pd"][3] == "python_feature_context"
    assert result.record.qwen_coverage == 1.0


def test_v8_revenue_does_not_use_other_income():
    from ml.agentic_ai.document_extraction import (
        DocumentExtractionResult,
        ExtractedPage,
    )
    from ml.agentic_ai.document_mapper import DeterministicDocumentMapper

    docs = DocumentExtractionResult(
        pages=[
            ExtractedPage(
                source_name="lk.pdf",
                page=1,
                method="pypdf",
                text=(
                    "Dalam Rp juta\n"
                    "Hasil Termijn Bersih 218.000\n"
                    "Total Pendapatan Lain-Lain 4.000\n"
                ),
            )
        ],
        warnings=[],
    )

    extraction = DeterministicDocumentMapper().extract(docs)
    assert extraction.raw_facts["revenue"].value == 218_000_000_000
    assert extraction.raw_facts["sales"].value == 218_000_000_000


def test_v9_rag_tool_schema():
    from ml.agentic_ai.schemas import POLICY_RAG_TOOL
    from ml.agentic_ai.tool_registry import definition

    spec = definition(POLICY_RAG_TOOL)
    params = spec["function"]["parameters"]

    assert spec["function"]["name"] == "query_credit_policy"
    assert "query" in params["properties"]
    assert "query" in params["required"]
    assert params["additionalProperties"] is False


def test_v9_rag_adapter_index_not_ready(monkeypatch):
    import sys
    import types

    fake_indeks = types.ModuleType("copilot.rag.indeks")
    fake_indeks.index_tersedia = lambda: False

    fake_pencarian = types.ModuleType("copilot.rag.pencarian")
    fake_pencarian.jawab = lambda *args, **kwargs: {
        "jawaban": "should not be called",
        "sitasi": [],
    }

    monkeypatch.setitem(sys.modules, "copilot.rag.indeks", fake_indeks)
    monkeypatch.setitem(sys.modules, "copilot.rag.pencarian", fake_pencarian)

    from ml.agentic_ai.rag_adapter import query_credit_policy

    out = query_credit_policy("Apa kebijakan kredit?")
    assert out["status"] == "index_not_ready"
    assert out["citations"] == []


def test_v9_agent_qwen_calls_rag_after_ml(monkeypatch):
    from types import SimpleNamespace
    import ml.agentic_ai.agent as agent_module
    from ml.agentic_ai.schemas import POLICY_RAG_TOOL
    from ml.agentic_ai.tool_registry import ToolTrace

    feature_context = {
        "pd": {"features": {"a": 1}, "missing_feature_names": []},
        "ews": {"features": {"b": 2}, "missing_feature_names": []},
        "lgd": {"features": {"c": 3}, "missing_feature_names": []},
        "pd_cluster": {"features": {"d": 4}, "missing_feature_names": []},
    }

    class FakeClient:
        def __init__(self):
            self.n = 0

        def chat(self, **kwargs):
            self.n += 1
            tool_names = [
                t["function"]["name"]
                for t in kwargs.get("tools", [])
            ]

            if "query_credit_policy" in tool_names:
                return {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "query_credit_policy",
                                "arguments": {
                                    "query": "Kebijakan kredit untuk fasilitas modal kerja",
                                    "top_k": 5,
                                },
                            }
                        }
                    ],
                }

            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "predict_pd", "arguments": {"run": True}}},
                    {"function": {"name": "predict_ews", "arguments": {"run": True}}},
                    {"function": {"name": "predict_lgd", "arguments": {"run": True}}},
                    {"function": {"name": "predict_pd_cluster", "arguments": {"run": True}}},
                ],
            }

    def fake_dispatch(
        name,
        arguments,
        *,
        execute=True,
        caller="unknown",
        llm_arguments=None,
        binding_source=None,
    ):
        if name == POLICY_RAG_TOOL:
            result = {
                "status": "retrieved",
                "query": arguments["query"],
                "answer": "Kebijakan tersedia (POJK X Pasal 1).",
                "citations": [
                    {
                        "rujukan": "POJK X Pasal 1",
                        "skor": 0.9,
                        "halaman": [1],
                    }
                ],
                "citation_count": 1,
            }
        else:
            result = {"status": "scored"}

        return ToolTrace(
            name=name,
            arguments=arguments,
            result=result,
            caller=caller,
            llm_arguments=llm_arguments,
            binding_source=binding_source,
        )

    monkeypatch.setattr(agent_module, "dispatch", fake_dispatch)

    settings = SimpleNamespace(
        qwen_agent_model="fake",
        max_tool_rounds=2,
        rag_tool_rounds=2,
        rag_enabled=True,
        agent_temperature=0,
    )

    result = agent_module.QwenMLAgent(
        client=FakeClient(),
        settings=settings,
    ).run(
        feature_context,
        policy_context={
            "borrower_name": "PT Test",
            "application_facts": {
                "facility_type": "KMK",
            },
        },
    )

    assert result.record.qwen_coverage == 1.0
    assert result.record.rag_qwen_attempted is True
    assert result.record.rag_retrieved is True
    assert result.record.qwen_agent_tool_coverage == 1.0
    assert result.record.rag_result["citation_count"] == 1


def test_v9_2_ml_llm_argument_policy():
    from ml.agentic_ai.tool_registry import ToolTrace

    good = ToolTrace(
        name="predict_pd",
        arguments={"features": {"x": 1}},
        llm_arguments={"run": True},
    )
    bad_empty = ToolTrace(
        name="predict_pd",
        arguments={"features": {"x": 1}},
        llm_arguments={},
    )
    bad_features = ToolTrace(
        name="predict_pd",
        arguments={"features": {"x": 1}},
        llm_arguments={"run": True, "features": {"x": 999}},
    )

    assert good.llm_argument_policy_compliant is True
    assert bad_empty.llm_argument_policy_compliant is False
    assert bad_features.llm_argument_policy_compliant is False


def test_v9_2_1_current_champion_names_configured():
    from ml.agentic_ai.config import MODEL_LAYOUT

    assert MODEL_LAYOUT["pd"]["champion"] == "pd_champion_new.joblib"
    assert MODEL_LAYOUT["ews"]["champion"] == "ews_xgboost_champion.joblib"
    assert MODEL_LAYOUT["lgd"]["champion"] == "final_lgd_xgboost_new.pkl"
    assert MODEL_LAYOUT["pd_cluster"]["champion"] == "pd_cluster_champion.joblib"


def test_v9_2_1_qwen_targeted_recovery_executes_missing_tools(monkeypatch):
    from ml.agentic_ai.agent import QwenMLAgent
    import ml.agentic_ai.agent as agent_module
    from ml.agentic_ai.schemas import REQUIRED_ML_TOOLS
    from ml.agentic_ai.tool_registry import ToolRecord, ToolTrace

    class FakeClient:
        def chat(self, *, model, messages, tools=None, temperature=0.0, **kwargs):
            tool_name = tools[0]["function"]["name"]
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": tool_name,
                            "arguments": {"run": True},
                        }
                    }
                ],
            }

    agent = QwenMLAgent(client=FakeClient())

    monkeypatch.setattr(
        agent_module,
        "dispatch",
        lambda name, arguments, **kwargs: ToolTrace(
            name=name,
            arguments=arguments,
            result={"status": "scored"},
            caller=kwargs.get("caller"),
            llm_arguments=kwargs.get("llm_arguments"),
            binding_source=kwargs.get("binding_source"),
        ),
    )

    record = ToolRecord()
    messages = []

    feature_context = {
        "pd": {"features": {"a": 1}},
        "ews": {"features": {"b": 1}},
        "lgd": {"features": {"c": 1}},
        "pd_cluster": {"features": {"d": 1}},
    }

    rounds = agent._qwen_recover_missing_ml_tools(
        record=record,
        messages=messages,
        feature_context=feature_context,
        execute_tools=True,
        model="qwen-test",
    )

    assert rounds == 4
    assert record.qwen_coverage == 1.0
    assert set(record.qwen_attempted_names) == set(REQUIRED_ML_TOOLS)


def test_v9_3_qwen_is_official_narrator():
    from ml.agentic_ai.config import Settings

    settings = Settings()
    assert settings.qwen_narrator_model
    assert settings.require_qwen_narrator() == settings.qwen_narrator_model
    assert settings.sahabat_model == settings.qwen_narrator_model
    assert settings.require_sahabat() == settings.qwen_narrator_model


def test_v9_3_legacy_narrator_alias():
    from ml.agentic_ai.narrator import QwenNarrator, SahabatNarrator

    assert SahabatNarrator is QwenNarrator


def test_v9_3_narrator_prompt_blocks_ood_as_risk_factor():
    from ml.agentic_ai.narrator import NARRATOR_SYSTEM

    assert "DATA QUALITY / DISTRIBUTION WARNING" in NARRATOR_SYSTEM
    assert "bukan otomatis risk factor" in NARRATOR_SYSTEM
    assert "Feature completeness rendah" in NARRATOR_SYSTEM
    assert "KBLI" in NARRATOR_SYSTEM


def test_v9_3_feature_completeness_is_deterministic_reporting():
    from ml.agentic_ai.eval.deepeval_v9 import evaluate_feature_completeness

    out = evaluate_feature_completeness(
        feature_context={
            "pd": {
                "observed_feature_count": 10,
                "expected_feature_count": 37,
                "feature_completeness_percent": 27.03,
            }
        },
        verbose=False,
    )

    assert out["layer"] == "input_feature_completeness"
    assert out["models"]["pd"]["feature_completeness_percent"] == 27.03
    assert out["interpretation_rule"] == "report_only_do_not_infer_accuracy"
