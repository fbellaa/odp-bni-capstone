from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .artifacts import ArtifactStore, FeatureDef
from .schemas import BorrowerExtraction, MODEL_KEYS


@dataclass(frozen=True)
class DerivedRule:
    dependencies: tuple[str, ...]
    function: Callable[[dict[str, float]], float]
    label: str


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        raise ZeroDivisionError("denominator is zero")
    return a / b


# Only deterministic, auditable formulas belong here. Extend this registry after
# matching it to the exact feature engineering used in each training notebook.
DERIVED_RULES: dict[str, DerivedRule] = {
    "der": DerivedRule(("interest_bearing_debt", "equity"), lambda x: _safe_div(x["interest_bearing_debt"], x["equity"]), "interest_bearing_debt / equity"),
    "debt_to_equity": DerivedRule(("interest_bearing_debt", "equity"), lambda x: _safe_div(x["interest_bearing_debt"], x["equity"]), "interest_bearing_debt / equity"),
    "debt_equity_ratio": DerivedRule(("interest_bearing_debt", "equity"), lambda x: _safe_div(x["interest_bearing_debt"], x["equity"]), "interest_bearing_debt / equity"),
    "ebitda_margin": DerivedRule(("ebitda", "sales"), lambda x: _safe_div(x["ebitda"], x["sales"]), "ebitda / sales"),
    "debt_to_ebitda": DerivedRule(("interest_bearing_debt", "ebitda"), lambda x: _safe_div(x["interest_bearing_debt"], x["ebitda"]), "interest_bearing_debt / ebitda"),
    "interest_coverage": DerivedRule(("ebitda", "interest_expense"), lambda x: _safe_div(x["ebitda"], x["interest_expense"]), "ebitda / interest_expense"),
    "icr": DerivedRule(("ebitda", "interest_expense"), lambda x: _safe_div(x["ebitda"], x["interest_expense"]), "ebitda / interest_expense"),
    "current_ratio": DerivedRule(("current_assets", "current_liabilities"), lambda x: _safe_div(x["current_assets"], x["current_liabilities"]), "current_assets / current_liabilities"),
    "net_profit_margin": DerivedRule(("net_income", "sales"), lambda x: _safe_div(x["net_income"], x["sales"]), "net_income / sales"),
    "roe": DerivedRule(("net_income", "equity"), lambda x: _safe_div(x["net_income"], x["equity"]), "net_income / equity"),
    "roa": DerivedRule(("net_income", "total_assets"), lambda x: _safe_div(x["net_income"], x["total_assets"]), "net_income / total_assets"),
    "collateral_coverage": DerivedRule(("collateral_value", "facility_limit"), lambda x: _safe_div(x["collateral_value"], x["facility_limit"]), "collateral_value / facility_limit"),
}

# Exact raw-fact aliases. These DO NOT invent values; they only translate the
# canonical borrower fact name into the exact feature column used by a model.
RAW_FACT_ALIASES: dict[str, tuple[str, ...]] = {
    "fin_total_aset_rp": ("total_assets",),
    "fin_total_liabilitas_rp": ("total_liabilities",),
    "fin_ekuitas_rp": ("equity", "total_equity"),
    "fin_penjualan_rp": ("sales", "revenue"),
    "app_penjualan_rp": ("sales", "revenue"),
    "fin_ebitda_rp": ("ebitda",),
    "fin_laba_bersih_rp": ("net_income", "net_profit"),
    "app_plafon_diminta_rp": ("requested_limit", "requested_amount", "requested_facility_limit"),
    "app_nilai_likuidasi_rp": ("liquidation_value", "collateral_liquidation_value"),
    "app_rating_internal": ("internal_rating",),
    "app_sektor_kbli": ("kbli_sector", "sector_kbli"),
    "app_skor_kredit": ("credit_score",),
    "app_tahun_berdiri": ("year_established", "founding_year"),
    "app_tenor_bulan": ("tenor_months", "tenor_bulan"),
    "app_umur_perusahaan_tahun": ("company_age_years",),
    "app_jenis_fasilitas": ("facility_type",),
    "app_nama_produk": ("product_name",),
    "app_revolving": ("revolving",),
    "app_ada_agunan_likuid": ("has_liquid_collateral",),
    "app_ada_jaminan_silang": ("has_cross_guarantee",),
    "app_jumlah_agunan": ("collateral_count",),
    "perilaku_dpd": ("days_past_due",),
    "perilaku_kolektibilitas": ("collectibility", "kol"),
    "perilaku_pemakaian_plafon": ("utilization", "facility_utilization"),
    "perilaku_restrukturisasi": ("restructured", "has_restructuring"),
    "graf_group_exposure_share": ("group_exposure_share",),
    "app_porsi_penjaminan": ("guarantee_share", "guarantee_portion"),
    "app_skala_pegawai": ("employee_scale",),
    "app_perusahaan_baru": ("new_company",),
    "app_dokumen_ringkas": ("document_summary_flag", "is_summary_document"),
    "perilaku_covenant_dilanggar": ("covenant_breached",),
}

# Safe, audit-friendly formulas whose names match the current model schema.
# More formulas should only be added after checking the ABT construction code.
PREFIXED_DERIVED_RULES: dict[str, DerivedRule] = {
    "fin_asset_turnover": DerivedRule(
        ("sales", "total_assets"),
        lambda x: _safe_div(x["sales"], x["total_assets"]),
        "sales / total_assets",
    ),
    "fin_current_ratio": DerivedRule(
        ("current_assets", "current_liabilities"),
        lambda x: _safe_div(x["current_assets"], x["current_liabilities"]),
        "current_assets / current_liabilities",
    ),
    "fin_quick_ratio": DerivedRule(
        ("current_assets", "inventory", "current_liabilities"),
        lambda x: _safe_div(x["current_assets"] - x["inventory"], x["current_liabilities"]),
        "(current_assets - inventory) / current_liabilities",
    ),
    "fin_der": DerivedRule(
        ("interest_bearing_debt", "equity"),
        lambda x: _safe_div(x["interest_bearing_debt"], x["equity"]),
        "interest_bearing_debt / equity",
    ),
    "fin_debt_to_ebitda": DerivedRule(
        ("interest_bearing_debt", "ebitda"),
        lambda x: _safe_div(x["interest_bearing_debt"], x["ebitda"]),
        "interest_bearing_debt / ebitda",
    ),
    "fin_gross_margin": DerivedRule(
        ("gross_profit", "sales"),
        lambda x: _safe_div(x["gross_profit"], x["sales"]),
        "gross_profit / sales",
    ),
    "fin_operating_margin": DerivedRule(
        ("operating_profit", "sales"),
        lambda x: _safe_div(x["operating_profit"], x["sales"]),
        "operating_profit / sales",
    ),
    "fin_icr": DerivedRule(
        ("ebitda", "interest_expense"),
        lambda x: _safe_div(x["ebitda"], x["interest_expense"]),
        "ebitda / interest_expense",
    ),
    "fin_roa": DerivedRule(
        ("net_income", "total_assets"),
        lambda x: _safe_div(x["net_income"], x["total_assets"]),
        "net_income / total_assets",
    ),
    "fin_re_to_ta": DerivedRule(
        ("retained_earnings", "total_assets"),
        lambda x: _safe_div(x["retained_earnings"], x["total_assets"]),
        "retained_earnings / total_assets",
    ),
    "fin_wc_to_ta": DerivedRule(
        ("current_assets", "current_liabilities", "total_assets"),
        lambda x: _safe_div(x["current_assets"] - x["current_liabilities"], x["total_assets"]),
        "(current_assets - current_liabilities) / total_assets",
    ),
    "fin_cfo_to_ebitda": DerivedRule(
        ("cfo", "ebitda"),
        lambda x: _safe_div(x["cfo"], x["ebitda"]),
        "cfo / ebitda",
    ),
    "fin_cfo_to_liability": DerivedRule(
        ("cfo", "total_liabilities"),
        lambda x: _safe_div(x["cfo"], x["total_liabilities"]),
        "cfo / total_liabilities",
    ),
    "fin_ebitda_negatif": DerivedRule(
        ("ebitda",),
        lambda x: float(x["ebitda"] < 0),
        "1 if ebitda < 0 else 0",
    ),
}



def _norm(name: str) -> str:
    return "_".join(str(name).strip().lower().replace("-", "_").replace("/", "_").split())


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip().replace("Rp", "").replace(" ", "")
        if not s:
            return None
        if "," in s:
            s = s.replace(".", "").replace(",", ".")
        elif s.count(".") > 1:
            s = s.replace(".", "")
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _coerce(value: Any, f: FeatureDef) -> Any:
    typ = f.json_type()
    if value is None:
        return None
    if typ == "string":
        return str(value)
    if typ == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "ya", "y"}
        return bool(value)
    num = _numeric(value)
    if num is None:
        return value
    if typ == "integer":
        return int(num)
    return float(num)


def _split_manual(manual: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manual = manual or {}
    raw: dict[str, Any] = {}
    per_model: dict[str, dict[str, Any]] = {k: {} for k in MODEL_KEYS}

    if isinstance(manual.get("raw_facts"), dict):
        raw.update(manual["raw_facts"])
    if isinstance(manual.get("model_features"), dict):
        for key in MODEL_KEYS:
            if isinstance(manual["model_features"].get(key), dict):
                per_model[key].update(manual["model_features"][key])

    # Convenient direct form: {"pd": {...}, "ews": {...}, ...}
    for key in MODEL_KEYS:
        if isinstance(manual.get(key), dict):
            per_model[key].update(manual[key])

    # Other top-level scalar values are treated as raw facts / potential exact feature names.
    reserved = {"raw_facts", "model_features", *MODEL_KEYS}
    for k, v in manual.items():
        if k not in reserved and not isinstance(v, dict):
            raw[k] = v
    return raw, per_model


class FeatureEngineer:
    """Build exact model inputs without allowing an LLM to perform calculations."""

    def __init__(self, store: ArtifactStore | None = None) -> None:
        self.store = store or ArtifactStore()

    def build(
        self,
        extraction: BorrowerExtraction,
        *,
        manual_input: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        manual_raw, manual_model = _split_manual(manual_input)

        facts: dict[str, Any] = {}
        fact_sources: dict[str, str] = {}
        for k, v in extraction.raw_facts.items():
            facts[_norm(k)] = v.value
            method = None
            if getattr(v, "evidence", None) is not None:
                method = getattr(v.evidence, "extraction_method", None)
            fact_sources[_norm(k)] = method or "structured_raw_fact"
        for k, v in manual_raw.items():
            facts[_norm(k)] = v
            fact_sources[_norm(k)] = "rm_manual_raw_fact"

        numeric_facts = {k: n for k, v in facts.items() if (n := _numeric(v)) is not None}
        report: dict[str, dict[str, Any]] = {}

        for model_key in MODEL_KEYS:
            defs = list(self.store.feature_defs(model_key))
            features: dict[str, Any] = {}
            provenance: dict[str, dict[str, Any]] = {}
            direct_extracted = extraction.direct_model_features.get(model_key, {})

            for f in defs:
                name = f.name
                norm = _norm(name)
                value = None
                source = None
                formula = None

                if name in manual_model.get(model_key, {}):
                    value = manual_model[model_key][name]
                    source = "rm_manual_model_feature"
                elif name in manual_raw:
                    value = manual_raw[name]
                    source = "rm_manual_exact_name"
                elif name in direct_extracted:
                    value = direct_extracted[name].value
                    source = "qwen_direct_model_feature"
                elif norm in facts:
                    value = facts[norm]
                    source = fact_sources.get(norm, "raw_fact_exact_name")
                elif name in RAW_FACT_ALIASES:
                    alias = next(
                        (a for a in RAW_FACT_ALIASES[name] if _norm(a) in facts),
                        None,
                    )
                    if alias is not None:
                        alias_norm = _norm(alias)
                        value = facts[alias_norm]
                        source = f"raw_fact_alias:{alias_norm}"
                if value is None and name in PREFIXED_DERIVED_RULES:
                    rule = PREFIXED_DERIVED_RULES[name]
                    if all(dep in numeric_facts for dep in rule.dependencies):
                        try:
                            value = rule.function(numeric_facts)
                            source = "deterministic_feature_engineering"
                            formula = rule.label
                        except ZeroDivisionError:
                            value = None
                if value is None and norm in DERIVED_RULES:
                    rule = DERIVED_RULES[norm]
                    if all(dep in numeric_facts for dep in rule.dependencies):
                        try:
                            value = rule.function(numeric_facts)
                            source = "deterministic_feature_engineering"
                            formula = rule.label
                        except ZeroDivisionError:
                            value = None

                if value is not None:
                    features[name] = _coerce(value, f)
                    provenance[name] = {
                        "source": source,
                        "formula": formula,
                    }

            expected = [f.name for f in defs]
            missing = [name for name in expected if name not in features]
            completeness = (len(features) / len(expected)) if expected else 0.0
            report[model_key] = {
                "features": features,
                "expected_feature_names": expected,
                "missing_feature_names": missing,
                "feature_provenance": provenance,
                "scorable_by_feature_presence": len(missing) == 0,
                "model_can_attempt_with_imputation": True,
                "observed_feature_count": len(features),
                "expected_feature_count": len(expected),
                "feature_completeness": round(completeness, 4),
                "feature_completeness_percent": round(completeness * 100, 2),
            }
        return report
