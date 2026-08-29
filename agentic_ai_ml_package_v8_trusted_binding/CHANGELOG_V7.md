# CHANGELOG V7

## Why V7 exists

V6 still depended on Qwen to map document wording into canonical raw facts. In real tests,
clean PDF text was available but feature coverage could remain near zero because the LLM
mapping was inconsistent.

## V7 architecture

`documents -> native/OCR/VLM -> deterministic document mapper -> deterministic feature engineering -> Qwen tool calling -> PD/EWS/LGD/PD Cluster -> verified results -> SahabatAI narrator`

### Key changes

- Added `document_mapper.py` as the primary document-to-fact extractor.
- Common Indonesian accounting and credit labels are mapped with auditable Python rules.
- Page monetary scales such as `Dalam Rp juta` are normalized to full IDR.
- Safe accounting components can be deterministically combined, with provenance retained.
- Added deterministic PD features including current ratio, quick ratio, asset turnover,
  DER, gross margin, operating margin, ROA, retained-earnings/TA, WC/TA and CFO ratios
  when their raw dependencies exist.
- Qwen structured extraction is now optional semantic fallback only.
- Qwen remains the ML tool-calling agent and is instructed to call all four mandatory tools.
- Tool traces now record whether a call came from `qwen` or `python_fallback`, making
  actual Qwen tool-calling coverage auditable.
- Missing borrower features are still padded as NaN by model tools and handled by the
  saved model preprocessing/imputer.

## Expected behavior for Sagara-style financial statements

Labels such as `TOTAL AKTIVA`, `Total Aktiva Lancar`, `Total Hutang Lancar`,
`Total Modal dan Laba`, `Hasil Termijn Bersih`, `LABA BRUTO`, `Biaya Bunga Bank`,
`LABA TAHUN BERJALAN`, `Total Setara Kas`, `Total Persediaan`, and
`Total Piutang Proyek` are mapped without an LLM.
