from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .document_extraction import DocumentExtractionResult


DEFAULT_KEYWORDS = (
    # identity / dates / units
    "pt ", "nama perusahaan", "debitur", "per ", "tahun yang berakhir", "dalam rp", "rp juta", "rp miliar",
    # balance sheet
    "total aktiva", "total aset", "aktiva lancar", "aset lancar", "hutang lancar", "liabilitas lancar",
    "hutang jangka panjang", "liabilitas jangka panjang", "total passiva", "total kewajiban", "total liabilitas",
    "modal", "ekuitas", "laba ditahan", "setara kas", "kas", "piutang", "persediaan",
    # income statement / cash flow
    "penjualan", "pendapatan", "termijn", "laba bruto", "laba kotor", "laba operasi", "laba usaha",
    "ebitda", "laba bersih", "laba tahun berjalan", "biaya bunga", "beban bunga", "arus kas", "cfo",
    # application / collateral
    "fasilitas", "plafon", "tenor", "agunan", "jaminan", "nilai likuidasi", "rating internal", "skor kredit",
    "kbli", "sektor", "revolving", "penjaminan", "pegawai", "tahun berdiri",
    # EWS / behavior
    "dpd", "days past due", "kolektibilitas", "restruktur", "covenant", "utilisasi", "pemakaian plafon",
    "rekening koran", "saldo", "tunggakan",
)


@dataclass(frozen=True)
class ReductionResult:
    text: str
    original_chars: int
    reduced_chars: int
    selected_lines: int
    source_count: int

    @property
    def reduction_ratio(self) -> float:
        if self.original_chars <= 0:
            return 0.0
        return self.reduced_chars / self.original_chars


def _feature_terms(feature_catalog: dict[str, list[dict[str, Any]]] | None) -> set[str]:
    terms: set[str] = set()
    for rows in (feature_catalog or {}).values():
        for item in rows:
            name = str(item.get("name") or "").strip().lower()
            if not name:
                continue
            # Strip modeling prefixes and split underscores to make matching more natural.
            for prefix in ("fin_", "app_", "perilaku_", "graf_"):
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            phrase = name.replace("_", " ").strip()
            if len(phrase) >= 4:
                terms.add(phrase)
    return terms


def reduce_documents_for_extraction(
    docs: DocumentExtractionResult,
    *,
    feature_catalog: dict[str, list[dict[str, Any]]] | None = None,
    max_chars: int = 14000,
    short_page_chars: int = 1800,
    neighbor_lines: int = 1,
) -> ReductionResult:
    """Create a compact, source-tagged extraction input for one Qwen call.

    Short pages are kept in full. For long pages we retain headers plus lines that
    match accounting/underwriting/EWS keywords and their neighboring lines. This is
    selection only: no values are calculated or altered.
    """

    dynamic_terms = _feature_terms(feature_catalog)
    keywords = tuple(DEFAULT_KEYWORDS) + tuple(dynamic_terms)
    blocks: list[str] = []
    selected_total = 0
    original_chars = len(docs.text)

    for page in docs.pages:
        raw = (page.text or "").strip()
        if not raw:
            continue
        marker = f"[SOURCE {page.source_name} | PAGE {page.page} | METHOD {page.method}]"

        if len(raw) <= short_page_chars:
            blocks.append(marker + "\n" + raw)
            selected_total += len([x for x in raw.splitlines() if x.strip()])
            continue

        lines = [x.strip() for x in raw.splitlines() if x.strip()]
        selected: set[int] = set(range(min(6, len(lines))))
        for i, line in enumerate(lines):
            lower = line.lower()
            if any(term in lower for term in keywords):
                for j in range(max(0, i - neighbor_lines), min(len(lines), i + neighbor_lines + 1)):
                    selected.add(j)

        if not selected:
            selected.update(range(min(12, len(lines))))

        chosen = [lines[i] for i in sorted(selected)]
        selected_total += len(chosen)
        blocks.append(marker + "\n" + "\n".join(chosen))

    text = "\n\n".join(blocks).strip()
    if len(text) > max_chars:
        # Keep the beginning and end. This preserves identity/units plus later credit/EWS sections.
        head = int(max_chars * 0.72)
        tail = max_chars - head
        text = text[:head].rstrip() + "\n\n[... REDUCED ...]\n\n" + text[-tail:].lstrip()

    return ReductionResult(
        text=text,
        original_chars=original_chars,
        reduced_chars=len(text),
        selected_lines=selected_total,
        source_count=len(docs.source_names),
    )
