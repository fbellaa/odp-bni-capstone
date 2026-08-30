from __future__ import annotations

import json
from typing import Any

from .agent import AgentResult
from .config import SETTINGS, Settings
from .ollama_client import OllamaClient
from .schemas import BorrowerExtraction, REQUIRED_ML_TOOLS


NARRATOR_SYSTEM = """\
Kamu adalah Qwen Credit Risk Narrator untuk HOLISTIC CREDIT RISK ASSESSMENT.
WAJIB menjawab dalam Bahasa Indonesia yang profesional, ringkas, dan mudah dipahami RM/analis.

Kamu HANYA boleh menggunakan fakta yang tersedia pada payload:
- borrower_extraction
- feature_status_by_model
- verified_tool_results
- tool_errors
- policy_rag_result
- document_warnings

ATURAN FAKTUAL KERAS:
1. Bahas hasil PD, EWS, LGD, dan PD Cluster secara terpisah.
2. Semua angka prediction, probability, threshold, LGD, risk band, cluster, atau skor
   WAJIB berasal dari verified_tool_results. Jangan membuat atau menghitung angka baru.
3. Jangan mengatakan debitur "pasti default" atau "tidak default".
   Gunakan bahasa probabilistik seperti "terindikasi", "diprediksi", atau "profil menyerupai".
4. Jangan mengklaim suatu feature MENYEBABKAN output model kecuali tersedia local explanation
   yang secara eksplisit mendukung kontribusi tersebut. Feature importance global bukan alasan individual.
5. Jangan menyimpulkan bahwa skor kredit tinggi, tenor tertentu, atau kode sektor tertentu otomatis
   membuat risiko lebih tinggi/rendah tanpa evidence lokal dari model.
6. Kode KBLI/sektor hanya boleh disebut sebagai identitas sektor. Jangan menyebut sektor "stabil",
   "aman", "berisiko", atau penilaian lain jika evidence tersebut tidak tersedia.
7. Nilai feature yang berada di luar reference range adalah DATA QUALITY / DISTRIBUTION WARNING,
   bukan otomatis risk factor. Jangan mengatakan nilai tinggi/rendah tersebut "concerning" secara risiko.
8. Jika ada kemungkinan scale/semantic mismatch (contoh skor 742 sedangkan reference model sekitar 0-100),
   jelaskan sebagai kemungkinan mismatch skala/pemetaan yang perlu divalidasi, bukan sebagai kesimpulan risiko.
9. Untuk rasio seperti ROA, jangan menebak apakah 1.0 berarti 1% atau 100%.
   Sebutkan perlunya validasi representasi desimal/persentase bila ada warning out-of-range.
10. Feature completeness rendah adalah metadata kualitas input. Jangan menyatakan bahwa prediksi
    "pasti tidak akurat" hanya karena completeness rendah. Katakan bahwa interpretasi perlu lebih hati-hati.
11. status=scored_with_imputation berarti model tetap menghasilkan output dan missing feature ditangani
    preprocessing/imputer tersimpan. Jangan menyebut model gagal hanya karena missing feature.
12. Hanya runtime/artifact error yang boleh disebut sebagai model failure.
13. Jika satu model gagal tetapi model lain sukses, berikan partial assessment dari model yang berhasil.

ATURAN POLICY / RAG:
14. Jika policy_rag_result tersedia, pernyataan kebijakan HANYA boleh berasal dari hasil RAG dan citations.
15. Jangan mengarang nama regulasi, Pasal, halaman, threshold, approval rule, atau ketentuan.
16. Bedakan "faktor penilaian yang diwajibkan kebijakan" dengan "threshold otomatis approve/reject".
    Jangan mengatakan kebijakan tidak memiliki kriteria sama sekali bila RAG menemukan faktor penilaian.
17. Jika RAG tidak menemukan threshold otomatis, katakan:
    "retrieval tidak menemukan threshold otomatis approve/reject pada corpus yang tersedia",
    bukan "regulasi tidak memiliki kriteria".
18. Jika index RAG belum siap atau retrieval gagal, katakan apa adanya.

ATURAN DECISION SUPPORT:
19. Output model dan narasi adalah decision support, bukan keputusan kredit final.
20. Jangan memberi keputusan approve/reject kecuali payload memang berisi decision rule terverifikasi.
21. Prioritaskan fakta model, warning input, policy evidence, dan hal yang perlu diverifikasi.

FORMAT WAJIB:
### Ringkasan Risiko Keseluruhan
### Probability of Default (PD)
### Early Warning System (EWS)
### Loss Given Default (LGD)
### PD Cluster / Profil Debitur
### Kualitas Data & Warning
### Kebijakan / Policy RAG
### Interpretasi untuk Analis/RM

Jangan membuat bagian "faktor pendorong model" jika tidak ada local explanation yang valid.
"""


class QwenNarrator:
    """Grounded Indonesian credit-risk narrator powered by Qwen."""

    def __init__(
        self,
        client: OllamaClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.s = settings or SETTINGS
        self.client = client or OllamaClient(self.s)

    def narrate(
        self,
        *,
        extraction: BorrowerExtraction,
        feature_context: dict[str, dict[str, Any]],
        agent: AgentResult,
        document_warnings: list[str] | None = None,
    ) -> str:
        model = self.s.require_qwen_narrator()
        verified = agent.record.last_success_by_name

        tool_errors = [
            {
                "tool": t.name,
                "error": t.error,
                "duration_ms": t.duration_ms,
            }
            for t in agent.record.traces
            if t.error and not t.duplicate_blocked
        ]

        payload = {
            "borrower_extraction": extraction.model_dump(mode="json"),
            "feature_status_by_model": {
                k: {
                    "missing_feature_names": v.get(
                        "missing_feature_names",
                        [],
                    ),
                    "model_can_attempt_with_imputation": v.get(
                        "model_can_attempt_with_imputation",
                        True,
                    ),
                    "observed_feature_count": v.get(
                        "observed_feature_count"
                    ),
                    "expected_feature_count": v.get(
                        "expected_feature_count"
                    ),
                    "feature_completeness_percent": v.get(
                        "feature_completeness_percent"
                    ),
                    "feature_provenance": v.get(
                        "feature_provenance",
                        {},
                    ),
                    "warnings": v.get(
                        "warnings",
                        [],
                    ),
                }
                for k, v in feature_context.items()
            },
            "required_tools": list(REQUIRED_ML_TOOLS),
            "verified_tool_results": {
                name: verified.get(name)
                for name in REQUIRED_ML_TOOLS
            },
            "tool_errors": tool_errors,
            "policy_rag_result": agent.record.rag_result,
            "policy_rag_attempted_by_qwen": (
                agent.record.rag_qwen_attempted
            ),
            "document_warnings": document_warnings or [],
        }

        reply = self.client.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": NARRATOR_SYSTEM,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        payload,
                        ensure_ascii=False,
                        default=str,
                    ),
                },
            ],
            temperature=self.s.narrator_temperature,
        )

        return (
            reply.get("content")
            or ""
        ).strip()


# Backward compatibility for older notebooks/imports.
SahabatNarrator = QwenNarrator
