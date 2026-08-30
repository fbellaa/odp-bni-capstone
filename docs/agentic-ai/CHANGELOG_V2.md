# V2 Change Log

- Mengubah arsitektur dari intent-driven model selection menjadi document-driven mandatory 4-model assessment.
- Menambahkan `document_extraction.py`.
- Mengganti `ParsedRequest`/intent parser dengan `BorrowerExtraction` + evidence.
- Menambahkan deterministic `feature_engineering.py`.
- Qwen sekarang wajib attempt PD, EWS, LGD, dan PD Cluster pada setiap borrower.
- Python memverifikasi mandatory tool coverage dan mengoreksi Qwen jika ada tool yang belum dipanggil.
- Tool dengan feature tidak lengkap mengembalikan `status=not_scorable` alih-alih di-skip.
- Narrator selalu membahas empat model dan missing information.
- DeepEval direvisi: Sahabat vs Qwen untuk extraction; Qwen vs Sahabat untuk tool calling.
- Notebook direvisi mengikuti flow upload documents → extract text → Sahabat → feature engineering → Qwen all tools → verified results → Sahabat narrator.
