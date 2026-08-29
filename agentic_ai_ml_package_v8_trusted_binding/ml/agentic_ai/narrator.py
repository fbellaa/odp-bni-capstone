from __future__ import annotations

import json
from typing import Any

from .agent import AgentResult
from .config import SETTINGS, Settings
from .ollama_client import OllamaClient
from .schemas import BorrowerExtraction, REQUIRED_ML_TOOLS

NARRATOR_SYSTEM = """\
Kamu adalah SahabatAI narrator untuk HOLISTIC CREDIT RISK ASSESSMENT dalam Bahasa Indonesia.
Kamu menerima fakta debitur, kualitas input, dan hasil tool ML yang telah diverifikasi Python.

ATURAN KERAS:
1. Bahas PD, EWS, LGD, dan PD Cluster.
2. Semua angka prediction/probability/threshold/LGD/cluster harus disalin dari verified_tool_results.
   Jangan menghitung atau membuat angka baru.
3. status=scored_with_imputation berarti model tetap menghasilkan output. Sebutkan
   completeness dan bahwa missing feature ditangani preprocessing/imputer tersimpan.
4. Jangan mengatakan model gagal hanya karena missing feature. Hanya runtime/artifact error
   yang boleh disebut sebagai model failure.
5. Jika explanation.scope=local_customer, sebut kontribusi model, bukan sebab kausal.
6. Jika explanation.scope=global_model, jelaskan itu importance global, bukan alasan individual.
7. Prediction adalah decision support, bukan keputusan kredit final.
8. Sebutkan warning dokumen/data bila ada.
9. Jangan mengklaim hasil dengan completeness sangat rendah sebagai bukti kuat tentang debitur.
10. Jangan membuat hubungan sebab-akibat dari feature importance, model margin, atau contribution.
    Gunakan bahasa 'berkontribusi pada output model' / 'terkait dengan output model', bukan 'menyebabkan'.
11. Jika tool gagal runtime, sebut tool/model itu belum tersedia dan lanjutkan assessment parsial
    menggunakan model yang berhasil. Jangan menyatakan seluruh holistic assessment gagal jika model lain sukses.

FORMAT:
- Ringkasan Risiko Keseluruhan
- Probability of Default (PD)
- Early Warning System (EWS)
- Loss Given Default (LGD)
- PD Cluster / Borrower Profile
- Faktor Pendorong Model
- Kualitas Data & Informasi yang Masih Kurang
- Interpretasi untuk Analis/RM
"""


class SahabatNarrator:
    def __init__(self, client: OllamaClient | None = None, settings: Settings | None = None) -> None:
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
        model = self.s.require_sahabat()
        verified = agent.record.last_success_by_name
        tool_errors = [
            {"tool": t.name, "error": t.error, "duration_ms": t.duration_ms}
            for t in agent.record.traces
            if t.error and not t.duplicate_blocked
        ]
        payload = {
            "borrower_extraction": extraction.model_dump(mode="json"),
            "feature_status_by_model": {
                k: {
                    "missing_feature_names": v.get("missing_feature_names", []),
                    "model_can_attempt_with_imputation": True,
                    "observed_feature_count": v.get("observed_feature_count"),
                    "expected_feature_count": v.get("expected_feature_count"),
                    "feature_completeness_percent": v.get("feature_completeness_percent"),
                    "feature_provenance": v.get("feature_provenance", {}),
                }
                for k, v in feature_context.items()
            },
            "required_tools": list(REQUIRED_ML_TOOLS),
            "verified_tool_results": {
                name: verified.get(name) for name in REQUIRED_ML_TOOLS
            },
            "tool_errors": tool_errors,
            "document_warnings": document_warnings or [],
        }
        reply = self.client.chat(
            model=model,
            messages=[
                {"role": "system", "content": NARRATOR_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
            ],
            temperature=self.s.narrator_temperature,
        )
        return (reply.get("content") or "").strip()
