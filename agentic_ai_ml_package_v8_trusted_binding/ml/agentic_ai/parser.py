from __future__ import annotations

import json
from typing import Any

from .config import SETTINGS, Settings
from .ollama_client import OllamaClient
from .schemas import BorrowerExtraction, MODEL_KEYS

EXTRACTOR_SYSTEM = """\
You are a structured commercial-credit document extractor. The source text may be
Indonesian or English. Your job is extraction, not credit analysis.

HARD RULES:
1. Output JSON that matches the supplied schema exactly.
2. Extract only facts explicitly supported by the source text or RM manual input.
3. Never calculate ratios, growth, aggregates, PD, EWS, LGD, clusters, or derived values.
4. Never guess missing values. Put uncertainty in missing_or_ambiguous.
5. Prefer canonical English snake_case raw_facts, e.g. sales, revenue, gross_profit,
   operating_profit, ebitda, net_income, interest_bearing_debt, equity,
   interest_expense, total_assets, total_liabilities, current_assets,
   current_liabilities, inventory, cash_and_equivalents, accounts_receivable,
   retained_earnings, cfo, collateral_value, liquidation_value, requested_limit,
   facility_limit, facility_type, internal_rating, kbli_sector, credit_score,
   founding_year, tenor_months, company_age_years, days_past_due, collectibility,
   utilization, restructured, covenant_breached, account_balance, transaction_count.
6. direct_model_features may contain a model feature only when the exact value is
   explicitly present in the source. If it needs calculation, leave it to Python.
7. Preserve evidence: source_document, page if marker exists, and a short quote.
8. Normalize source monetary units to full IDR only when the document explicitly gives
   a scale. Example: 366.000 in a section labelled 'Dalam Rp juta' means
   366000000000 IDR. Do not infer an unstated scale.
9. Map clear Indonesian accounting labels to canonical facts, e.g. TOTAL AKTIVA ->
   total_assets; TOTAL HUTANG LANCAR -> current_liabilities; LABA TAHUN BERJALAN ->
   net_income.
10. Do not produce recommendations or a narrative assessment.
"""


class QwenStructuredExtractor:
    """Optional semantic fallback extractor. V7 primary mapping is deterministic Python."""

    def __init__(self, client: OllamaClient | None = None, settings: Settings | None = None) -> None:
        self.s = settings or SETTINGS
        self.client = client or OllamaClient(self.s)

    @staticmethod
    def _catalog_text(feature_catalog: dict[str, list[dict[str, Any]]] | None) -> str:
        feature_catalog = feature_catalog or {}
        rows: list[str] = []
        for key in MODEL_KEYS:
            names = [str(f.get("name")) for f in feature_catalog.get(key, []) if f.get("name")]
            if names:
                rows.append(f"[{key}] " + ", ".join(names))
        return "\n".join(rows) or "(catalog omitted; extract canonical raw facts)"

    def extract(
        self,
        document_text: str,
        *,
        feature_catalog: dict[str, list[dict[str, Any]]] | None = None,
        manual_input: dict[str, Any] | None = None,
        model_override: str | None = None,
    ) -> BorrowerExtraction:
        model = model_override or self.s.qwen_extractor_model
        manual_json = json.dumps(manual_input or {}, ensure_ascii=False, default=str)
        user_payload = (
            "MODEL FEATURE NAMES (reference only; do not calculate):\n"
            + self._catalog_text(feature_catalog)
            + "\n\nRM MANUAL INPUT:\n"
            + manual_json
            + "\n\nDOCUMENT EXCERPTS WITH SOURCE MARKERS:\n"
            + (document_text or "(no document text)")
        )
        return self.client.structured(
            model=model,
            messages=[
                {"role": "system", "content": EXTRACTOR_SYSTEM},
                {"role": "user", "content": user_payload},
            ],
            schema=BorrowerExtraction,
            temperature=self.s.extractor_temperature,
            retries=2,
        )

    parse = extract


# Backward compatibility for code that imported the old class name.
SahabatExtractor = QwenStructuredExtractor
