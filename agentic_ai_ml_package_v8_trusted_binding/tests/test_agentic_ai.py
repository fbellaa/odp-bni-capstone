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


def test_v8_tool_schema_has_no_business_arguments():
    from ml.agentic_ai.tool_registry import definition

    spec = definition("predict_pd")
    params = spec["function"]["parameters"]
    assert params["properties"] == {}
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
                            "arguments": {"features": {"fin_current_ratio": 999}},
                        }
                    },
                    {
                        "function": {
                            "name": "predict_ews",
                            "arguments": {"features": {"wrong": 999}},
                        }
                    },
                    {
                        "function": {
                            "name": "predict_lgd",
                            "arguments": {"features": {"wrong": 999}},
                        }
                    },
                    {
                        "function": {
                            "name": "predict_pd_cluster",
                            "arguments": {"features": {"app_skor_kredit": 999}},
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
