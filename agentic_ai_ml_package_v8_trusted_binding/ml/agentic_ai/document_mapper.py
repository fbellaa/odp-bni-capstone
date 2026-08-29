from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any, Callable

from .document_extraction import DocumentExtractionResult, ExtractedPage
from .schemas import BorrowerExtraction, Evidence, ExtractedValue, MODEL_KEYS


@dataclass(frozen=True)
class MappingCandidate:
    fact: str
    value: Any
    unit: str | None
    source_document: str
    page: int | None
    quote: str
    score: int
    formula: str | None = None


@dataclass(frozen=True)
class NumericRule:
    fact: str
    patterns: tuple[str, ...]
    value_kind: str = "money"  # money | number | integer | percent
    priority: int = 100


@dataclass(frozen=True)
class TextRule:
    fact: str
    patterns: tuple[str, ...]
    priority: int = 100


# These rules intentionally match human document labels, not model column names.
# The mapping is auditable and deterministic. Add synonyms here when new document
# templates are encountered; do not make the LLM responsible for model-feature math.
NUMERIC_RULES: tuple[NumericRule, ...] = (
    NumericRule("current_assets", (r"total\s+(?:aktiva|aset)\s+lancar",), priority=180),
    NumericRule("current_liabilities", (r"total\s+(?:hutang|liabilitas|kewajiban)\s+lancar",), priority=180),
    NumericRule("long_term_liabilities", (r"total\s+(?:hutang|liabilitas|kewajiban)\s+jangka\s+panjang",), priority=180),
    NumericRule("total_assets", (r"total\s+(?:aktiva|aset)(?!\s+lancar|\s+tetap|\s+lain)",), priority=170),
    NumericRule("total_liabilities", (r"total\s+(?:liabilitas|kewajiban)(?!\s+lancar|\s+jangka)",), priority=170),
    NumericRule("equity", (r"total\s+modal\s+dan\s+laba", r"total\s+ekuitas", r"total\s+modal$"), priority=175),
    NumericRule("retained_earnings", (r"laba\s+ditahan",), priority=160),
    NumericRule("cash_and_equivalents", (r"total\s+setara\s+kas", r"^setara\s+kas$"), priority=160),
    NumericRule("inventory", (r"total\s+persediaan", r"^persediaan$"), priority=160),
    NumericRule("accounts_receivable", (r"total\s+piutang\s+proyek", r"total\s+piutang\s+usaha", r"^piutang\s+usaha$"), priority=160),
    NumericRule("revenue", (r"hasil\s+termijn\s+bersih", r"penjualan\s+bersih", r"pendapatan\s+usaha\s+bersih", r"total\s+pendapatan(?:\s+usaha)?(?!\s+lain)"), priority=180),
    NumericRule("gross_profit", (r"laba\s+(?:bruto|kotor)",), priority=170),
    NumericRule("operating_profit", (r"laba\s+(?:operasi|usaha)",), priority=170),
    NumericRule("net_income", (r"laba\s+tahun\s+berjalan", r"laba\s+bersih"), priority=175),
    NumericRule("profit_before_tax", (r"laba\s+sebelum\s+pajak",), priority=150),
    NumericRule("interest_expense", (r"biaya\s+bunga\s+bank", r"beban\s+bunga"), priority=170),
    NumericRule("cfo", (r"arus\s+kas\s+(?:bersih\s+)?(?:dari|untuk)\s+aktivitas\s+operasi", r"cash\s+flow\s+from\s+operations", r"\bcfo\b"), priority=160),
    NumericRule("cogs", (r"harga\s+pokok\s+penjualan", r"hpp\s+penjualan", r"total\s+biaya\s+proyek\s+langsung"), priority=150),
    NumericRule("indirect_project_expense", (r"total\s+biaya\s+proyek\s+tidak\s+langsung",), priority=150),
    NumericRule("admin_general_expense", (r"total\s+biaya\s+administrasi\s+dan\s+umum",), priority=150),
    NumericRule("short_term_bank_debt", (r"total\s+hutang\s+bank\s+lancar", r"^hutang\s+bank$"), priority=160),
    NumericRule("long_term_bank_debt", (r"^hutang\s+bank\s+jangka\s+panjang$",), priority=165),
    NumericRule("consumer_financing_debt", (r"^hutang\s+pembiayaan\s+konsumen$",), priority=160),
    NumericRule("requested_limit", (r"plafon\s+(?:diminta|permohonan|pengajuan)", r"requested\s+limit"), priority=170),
    NumericRule("liquidation_value", (r"nilai\s+likuidasi",), priority=170),
    NumericRule("credit_score", (r"skor\s+kredit", r"credit\s+score"), value_kind="number", priority=170),
    NumericRule("tenor_months", (r"tenor(?:\s+bulan)?",), value_kind="integer", priority=160),
    NumericRule("founding_year", (r"tahun\s+berdiri",), value_kind="integer", priority=160),
    NumericRule("days_past_due", (r"\bdpd\b", r"days\s+past\s+due"), value_kind="integer", priority=160),
    NumericRule("collectibility", (r"kolektibilitas", r"\bkol\b"), value_kind="integer", priority=160),
    NumericRule("utilization", (r"pemakaian\s+plafon", r"utilisasi"), value_kind="percent", priority=150),
    NumericRule("guarantee_share", (r"porsi\s+penjaminan",), value_kind="percent", priority=150),
)

TEXT_RULES: tuple[TextRule, ...] = (
    TextRule("internal_rating", (r"rating\s+internal",), priority=170),
    TextRule("kbli_sector", (r"sektor\s+kbli", r"\bkbli\b"), priority=165),
    TextRule("facility_type", (r"jenis\s+fasilitas", r"fasilitas\s+kredit"), priority=165),
    TextRule("product_name", (r"nama\s+produk",), priority=150),
    TextRule("employee_scale", (r"skala\s+pegawai",), priority=150),
)

BOOL_RULES: dict[str, tuple[str, ...]] = {
    "revolving": (r"\brevolving\b",),
    "has_liquid_collateral": (r"agunan\s+likuid",),
    "has_cross_guarantee": (r"jaminan\s+silang",),
    "restructured": (r"restrukturisasi", r"restruktur"),
    "covenant_breached": (r"covenant\s+(?:dilanggar|breach)", r"pelanggaran\s+covenant"),
    "new_company": (r"perusahaan\s+baru",),
}


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _page_money_scale(text: str) -> float:
    lower = (text or "").lower()
    if re.search(r"dalam\s+rp\s*(?:triliun|tn)\b", lower):
        return 1e12
    if re.search(r"dalam\s+rp\s*(?:miliar|milyar|bn)\b", lower):
        return 1e9
    if re.search(r"dalam\s+rp\s*(?:juta|mn)\b", lower):
        return 1e6
    if re.search(r"dalam\s+rp\s*(?:ribu|rb)\b", lower):
        return 1e3
    return 1.0


def _explicit_money_scale(line: str) -> float | None:
    lower = line.lower()
    if re.search(r"\b(?:triliun|tn)\b", lower):
        return 1e12
    if re.search(r"\b(?:miliar|milyar|bn)\b", lower):
        return 1e9
    if re.search(r"\b(?:juta|mn)\b", lower):
        return 1e6
    if re.search(r"\b(?:ribu|rb)\b", lower):
        return 1e3
    return None


def _parse_id_number(token: str) -> float | None:
    """Parse common Indonesian/English numeric notation without guessing units."""
    s = str(token or "").strip()
    s = re.sub(r"(?i)rp\.?", "", s)
    s = s.replace("%", "").replace(" ", "")
    s = re.sub(r"[^0-9,\.\-+]", "", s)
    if not s or s in {"-", "+"}:
        return None

    # 366.000 / 13.500: Indonesian thousands grouping.
    if re.fullmatch(r"[-+]?\d{1,3}(?:\.\d{3})+", s):
        s = s.replace(".", "")
    # 1,234,567: English thousands grouping.
    elif re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+", s):
        s = s.replace(",", "")
    # 13.5 or 13,5 decimal.
    elif "," in s and "." not in s:
        s = s.replace(",", ".")
    elif "," in s and "." in s:
        # Indonesian mixed: 1.234,56
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _numeric_tail(line: str) -> tuple[str, float] | None:
    # Prefer the final numeric token because financial statement lines are typically label -> value.
    matches = list(re.finditer(r"[-+]?\d[\d\.,]*\s*%?", line))
    if not matches:
        return None
    m = matches[-1]
    value = _parse_id_number(m.group(0))
    if value is None:
        return None
    return m.group(0), value


def _money_value(line: str, page_scale: float) -> float | None:
    parsed = _numeric_tail(line)
    if parsed is None:
        return None
    _, value = parsed
    scale = _explicit_money_scale(line)
    if scale is None:
        scale = page_scale
    return float(value) * float(scale)


def _percent_value(line: str) -> float | None:
    parsed = _numeric_tail(line)
    if parsed is None:
        return None
    token, value = parsed
    # Normalize explicit percentages to proportions; plain utilization numbers are kept as-is.
    if "%" in token or "%" in line:
        return float(value) / 100.0
    return float(value)


def _label_matches(line: str, pattern: str) -> bool:
    # Match the label in the prefix before the final numeric token, to avoid matching narrative amounts.
    parsed = _numeric_tail(line)
    prefix = line
    if parsed is not None:
        token, _ = parsed
        idx = line.rfind(token)
        if idx >= 0:
            prefix = line[:idx]
    prefix = prefix.strip()
    return re.search(pattern, prefix, flags=re.IGNORECASE) is not None


def _candidate_score(line: str, base: int) -> int:
    lower = line.lower().strip()
    score = base
    if lower.startswith("total ") or " total " in f" {lower} ":
        score += 15
    if ":" in line:
        score += 2
    return score


def _text_after_label(line: str, pattern: str) -> str | None:
    m = re.search(pattern, line, flags=re.IGNORECASE)
    if not m:
        return None
    tail = line[m.end():].strip(" :-\t")
    if not tail:
        return None
    return tail



def _clean_text_fact(fact: str, value: str) -> str | None:
    """Conservative cleanup for document text labels.

    The mapper prefers missing over a malformed categorical value.
    """
    value = _normalize_spaces(value).strip("|:;,- ")
    if not value:
        return None

    if fact == "kbli_sector":
        # KBLI is normally a 5-digit code. Keep only the code when a narrative
        # sentence follows it, e.g. "42919. Berdasarkan penilaian..."
        m = re.search(r"\b(\d{5})\b", value)
        if m:
            return m.group(1)

    if fact == "facility_type":
        # Reject obvious date/header fragments accidentally captured as a value.
        if re.search(
            r"(?i)\b\d{1,2}\s+(?:januari|februari|maret|april|mei|juni|juli|"
            r"agustus|september|oktober|november|desember)\s+\d{4}\b",
            value,
        ):
            return None
        if value.startswith("|"):
            return None

    # Avoid swallowing long narrative paragraphs into categorical features.
    if len(value) > 100:
        value = value.split(".", 1)[0].strip()

    return value or None


def _bool_from_line(line: str) -> bool | None:
    lower = line.lower()
    negative = ("tidak" in lower or "no" in lower or "false" in lower or "tidak ada" in lower)
    positive = ("ya" in lower or "yes" in lower or "true" in lower or "ada" in lower or "dilanggar" in lower or "breach" in lower)
    if negative:
        return False
    if positive:
        return True
    # Literal 1/0 at end.
    parsed = _numeric_tail(line)
    if parsed:
        _, value = parsed
        if value in {0, 1}:
            return bool(value)
    return None


def _make_value(candidate: MappingCandidate) -> ExtractedValue:
    return ExtractedValue(
        value=candidate.value,
        unit=candidate.unit,
        explicit_in_source=candidate.formula is None,
        confidence=0.99 if candidate.formula is None else 1.0,
        evidence=Evidence(
            source_document=candidate.source_document,
            page=candidate.page,
            quote=candidate.quote[:500],
            extraction_method=("deterministic_formula" if candidate.formula else "deterministic_document_mapper"),
        ),
    )


def _best(candidates: list[MappingCandidate]) -> MappingCandidate | None:
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: (x.score, -(x.page or 10_000)), reverse=True)[0]


class DeterministicDocumentMapper:
    """Fast primary extractor for common Indonesian commercial-credit documents.

    It reads already-extracted page text, maps supported document labels to canonical
    raw facts, preserves provenance, and performs only explicitly declared safe
    arithmetic (e.g. current liabilities + long-term liabilities). It never predicts
    risk and never calls an LLM.
    """

    def extract(self, docs: DocumentExtractionResult) -> BorrowerExtraction:
        candidates: dict[str, list[MappingCandidate]] = {}
        pt_names: list[str] = []
        dates: list[tuple[str, str, int | None]] = []

        for page in docs.pages:
            text = page.text or ""
            page_scale = _page_money_scale(text)
            lines = [_normalize_spaces(x) for x in text.splitlines() if _normalize_spaces(x)]

            # Borrower name: most frequently repeated uppercase PT name usually identifies the report owner.
            for line in lines[:15]:
                for m in re.finditer(r"\bPT\s+[A-Z][A-Z0-9 .&\-/]{3,80}", line):
                    name = _normalize_spaces(m.group(0)).rstrip(" .,-")
                    if len(name) <= 90:
                        pt_names.append(name)

            for line in lines[:20]:
                dm = re.search(
                    r"(?i)\b(?:per|berakhir|tanggal)\s+(\d{1,2}\s+(?:januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4})",
                    line,
                )
                if dm:
                    dates.append((_normalize_spaces(dm.group(1)), page.source_name, page.page))

            for rule in NUMERIC_RULES:
                for pattern in rule.patterns:
                    for line in lines:
                        if not _label_matches(line, pattern):
                            continue
                        if rule.value_kind == "money":
                            value = _money_value(line, page_scale)
                            unit = "IDR"
                        elif rule.value_kind == "percent":
                            value = _percent_value(line)
                            unit = "ratio" if value is not None and ("%" in line) else None
                        else:
                            parsed = _numeric_tail(line)
                            value = None if parsed is None else parsed[1]
                            unit = None
                            if value is not None and rule.value_kind == "integer":
                                value = int(round(value))
                        if value is None:
                            continue
                        candidates.setdefault(rule.fact, []).append(
                            MappingCandidate(
                                fact=rule.fact,
                                value=value,
                                unit=unit,
                                source_document=page.source_name,
                                page=page.page,
                                quote=line,
                                score=_candidate_score(line, rule.priority),
                            )
                        )
                        break

            for rule in TEXT_RULES:
                for pattern in rule.patterns:
                    for line in lines:
                        if re.search(pattern, line, flags=re.IGNORECASE) is None:
                            continue
                        value = _text_after_label(line, pattern)
                        value = _clean_text_fact(rule.fact, value) if value else None
                        if value:
                            candidates.setdefault(rule.fact, []).append(
                                MappingCandidate(
                                    fact=rule.fact,
                                    value=value,
                                    unit=None,
                                    source_document=page.source_name,
                                    page=page.page,
                                    quote=line,
                                    score=_candidate_score(line, rule.priority),
                                )
                            )
                            break

            for fact, patterns in BOOL_RULES.items():
                for pattern in patterns:
                    for line in lines:
                        if re.search(pattern, line, flags=re.IGNORECASE) is None:
                            continue
                        value = _bool_from_line(line)
                        if value is not None:
                            candidates.setdefault(fact, []).append(
                                MappingCandidate(
                                    fact=fact,
                                    value=value,
                                    unit=None,
                                    source_document=page.source_name,
                                    page=page.page,
                                    quote=line,
                                    score=140,
                                )
                            )
                            break

        raw_facts: dict[str, ExtractedValue] = {}
        for fact, rows in candidates.items():
            winner = _best(rows)
            if winner is not None:
                raw_facts[fact] = _make_value(winner)

        # Choose the most repeated company header, not a random affiliate mentioned once.
        borrower_name: str | None = None
        if pt_names:
            borrower_name = Counter(pt_names).most_common(1)[0][0]
            raw_facts.setdefault(
                "company_name",
                ExtractedValue(
                    value=borrower_name,
                    unit=None,
                    explicit_in_source=True,
                    confidence=0.99,
                    evidence=Evidence(extraction_method="deterministic_document_mapper"),
                ),
            )
        if dates:
            date, src, page = Counter([d[0] for d in dates]).most_common(1)[0][0], dates[0][1], dates[0][2]
            raw_facts.setdefault(
                "report_date",
                ExtractedValue(
                    value=date,
                    explicit_in_source=True,
                    confidence=0.98,
                    evidence=Evidence(source_document=src, page=page, extraction_method="deterministic_document_mapper"),
                ),
            )

        # Canonical sales alias. In financial statements, revenue/penjualan is the
        # model's sales base unless a distinct sales line is explicitly available.
        if "sales" not in raw_facts and "revenue" in raw_facts:
            src = raw_facts["revenue"]
            raw_facts["sales"] = ExtractedValue(
                value=src.value,
                unit=src.unit,
                explicit_in_source=src.explicit_in_source,
                confidence=src.confidence,
                evidence=Evidence(
                    source_document=(src.evidence.source_document if src.evidence else None),
                    page=(src.evidence.page if src.evidence else None),
                    quote=(src.evidence.quote if src.evidence else "revenue -> sales canonical alias"),
                    extraction_method="deterministic_alias",
                ),
            )

        # Safe arithmetic on explicitly extracted accounting components.
        self._derive(raw_facts)

        return BorrowerExtraction(
            borrower_name=borrower_name,
            raw_facts=raw_facts,
            direct_model_features={k: {} for k in MODEL_KEYS},
            missing_or_ambiguous=[],
            notes=[
                "Primary extraction used deterministic label mapping. Qwen is not used for model-feature calculations."
            ],
        )

    @staticmethod
    def _derive(raw_facts: dict[str, ExtractedValue]) -> None:
        def num(name: str) -> float | None:
            if name not in raw_facts:
                return None
            try:
                return float(raw_facts[name].value)
            except (TypeError, ValueError):
                return None

        # Balance-sheet identity when a total-liability line is absent.
        if "total_liabilities" not in raw_facts:
            ca = num("current_liabilities")
            lt = num("long_term_liabilities")
            if ca is not None and lt is not None:
                raw_facts["total_liabilities"] = ExtractedValue(
                    value=ca + lt,
                    unit="IDR",
                    explicit_in_source=False,
                    confidence=1.0,
                    evidence=Evidence(
                        quote="current_liabilities + long_term_liabilities",
                        extraction_method="deterministic_formula",
                    ),
                )

        # Explicitly labelled bank/consumer-financing debt is treated as interest-bearing debt.
        debt_parts = [
            num("short_term_bank_debt"),
            num("long_term_bank_debt"),
            num("consumer_financing_debt"),
        ]
        present = [x for x in debt_parts if x is not None]
        if "interest_bearing_debt" not in raw_facts and present:
            raw_facts["interest_bearing_debt"] = ExtractedValue(
                value=sum(present),
                unit="IDR",
                explicit_in_source=False,
                confidence=1.0,
                evidence=Evidence(
                    quote="sum(explicit bank debt + consumer financing debt components)",
                    extraction_method="deterministic_formula",
                ),
            )

        # Use gross profit less explicitly stated indirect + admin expenses as operating profit
        # only when operating profit is not explicitly reported.
        if "operating_profit" not in raw_facts:
            gp = num("gross_profit")
            indirect = num("indirect_project_expense")
            admin = num("admin_general_expense")
            if gp is not None and indirect is not None and admin is not None:
                raw_facts["operating_profit"] = ExtractedValue(
                    value=gp - indirect - admin,
                    unit="IDR",
                    explicit_in_source=False,
                    confidence=1.0,
                    evidence=Evidence(
                        quote="gross_profit - indirect_project_expense - admin_general_expense",
                        extraction_method="deterministic_formula",
                    ),
                )


def merge_extractions(
    primary: BorrowerExtraction,
    fallback: BorrowerExtraction | None,
) -> BorrowerExtraction:
    """Merge an optional semantic fallback without overriding deterministic facts."""
    if fallback is None:
        return primary

    raw = dict(primary.raw_facts)
    for key, value in fallback.raw_facts.items():
        raw.setdefault(key, value)

    direct = {k: dict(primary.direct_model_features.get(k, {})) for k in MODEL_KEYS}
    for model_key in MODEL_KEYS:
        for key, value in fallback.direct_model_features.get(model_key, {}).items():
            direct[model_key].setdefault(key, value)

    return BorrowerExtraction(
        borrower_name=primary.borrower_name or fallback.borrower_name,
        raw_facts=raw,
        direct_model_features=direct,
        missing_or_ambiguous=list(dict.fromkeys(primary.missing_or_ambiguous + fallback.missing_or_ambiguous)),
        notes=list(dict.fromkeys(primary.notes + fallback.notes)),
    )
