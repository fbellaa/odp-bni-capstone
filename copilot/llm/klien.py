"""Klien Ollama - satu-satunya tempat kode ini bicara ke model.

Memakai HTTP API Ollama langsung lewat `requests`, bukan paket `ollama`.
Alasannya praktis: `requests` sudah jadi dependensi antarmuka, dan di Kaggle /
Colab satu dependensi yang tidak perlu dipasang berarti satu kegagalan yang
tidak perlu terjadi di menit pertama demo.

Empat bentuk pemanggilan yang dipakai lapisan di atasnya:

    chat()          percakapan biasa, opsional dengan tool
    chat_arus()     versi streaming untuk chatbox
    terstruktur()   keluaran JSON yang divalidasi ke model Pydantic
    embed()         vektor untuk index RAG
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator, Sequence, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from copilot.konfigurasi import ANGGARAN_GB, PENGATURAN, Pengaturan

LOG = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class GalatOllama(RuntimeError):
    """Server Ollama tidak bisa dihubungi atau menolak permintaan."""


class ModelTidakAda(GalatOllama):
    """Model yang diminta belum ditarik ke server Ollama."""


class KlienOllama:
    def __init__(self, pengaturan: Pengaturan | None = None) -> None:
        self.p = pengaturan or PENGATURAN
        self.host = self.p.host_ollama.rstrip("/")
        self._sesi = requests.Session()

    # ------------------------------------------------------------- kesehatan
    def hidup(self) -> bool:
        try:
            self._sesi.get(f"{self.host}/api/tags", timeout=5).raise_for_status()
            return True
        except requests.RequestException:
            return False

    def daftar_model(self) -> list[str]:
        try:
            r = self._sesi.get(f"{self.host}/api/tags", timeout=10)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise GalatOllama(
                f"Tidak bisa menghubungi Ollama di {self.host}. "
                "Jalankan `ollama serve` dulu (di Kaggle/Colab: lihat "
                "copilot/scripts/siapkan_ollama.sh)."
            ) from exc
        return [m["name"] for m in r.json().get("models", [])]

    def pastikan_model(self, nama: str) -> None:
        """Gagal lebih awal dengan pesan yang bisa ditindaklanjuti.

        Ollama sendiri hanya menjawab 404 dengan teks pendek; di tengah demo
        itu tidak cukup untuk tahu tag mana yang sebenarnya terpasang.
        """
        tersedia = self.daftar_model()
        # Ollama menyimpan tag lengkap ("qwen2.5:3b-instruct"); pemanggil kerap
        # menulis tanpa tag, yang berarti ":latest".
        target = nama if ":" in nama else f"{nama}:latest"
        if target in tersedia or nama in tersedia:
            return
        terpasang = ", ".join(tersedia) or "(kosong)"
        raise ModelTidakAda(
            f"Model {nama!r} belum ada di Ollama.\n"
            f"  tarik dengan  : ollama pull {nama}\n"
            f"  yang terpasang: {terpasang}"
        )

    # ------------------------------------------------------------------ chat
    def chat(
        self,
        pesan: list[dict[str, Any]],
        *,
        peran: str = "chat",
        alat: Sequence[dict] | None = None,
        suhu: float | None = None,
        format_json: bool = False,
    ) -> dict[str, Any]:
        """Satu putaran /api/chat. Mengembalikan objek `message` mentah.

        Objek itu dipertahankan apa adanya karena lapisan agen membutuhkan
        `tool_calls` di dalamnya - membungkusnya jadi dataclass hanya akan
        memaksa membongkarnya lagi.
        """
        model = self.p.model_untuk(peran)
        muatan: dict[str, Any] = {
            "model": model,
            "messages": pesan,
            "stream": False,
            "options": {"temperature": self._suhu(peran) if suhu is None else suhu},
        }
        if alat:
            muatan["tools"] = list(alat)
        if format_json:
            muatan["format"] = "json"
        return self._panggil("/api/chat", muatan, model)["message"]

    def chat_arus(
        self,
        pesan: list[dict[str, Any]],
        *,
        peran: str = "chat",
        suhu: float | None = None,
    ) -> Iterator[str]:
        """Versi streaming untuk chatbox - potongan teks, bukan objek pesan."""
        model = self.p.model_untuk(peran)
        muatan = {
            "model": model,
            "messages": pesan,
            "stream": True,
            "options": {"temperature": self._suhu(peran) if suhu is None else suhu},
        }
        try:
            with self._sesi.post(
                f"{self.host}/api/chat", json=muatan, stream=True, timeout=self.p.timeout
            ) as r:
                self._periksa(r, model)
                for baris in r.iter_lines():
                    if not baris:
                        continue
                    potongan = json.loads(baris)
                    isi = potongan.get("message", {}).get("content", "")
                    if isi:
                        yield isi
                    if potongan.get("done"):
                        break
        except requests.RequestException as exc:
            raise GalatOllama(f"Aliran chat ke {model} terputus: {exc}") from exc

    # ----------------------------------------------------------- terstruktur
    def terstruktur(
        self,
        pesan: list[dict[str, Any]],
        skema: type[T],
        *,
        peran: str = "ekstraksi",
        percobaan: int = 2,
    ) -> T:
        """Minta keluaran JSON dan validasi ke `skema`.

        Model 3B kerap menghasilkan JSON yang bentuknya hampir benar - satu
        field bernama lain, angka dikirim sebagai string. Percobaan kedua
        mengirim balik pesan galat Pydantic apa adanya; itu jauh lebih efektif
        daripada memperkeras prompt awal.
        """
        percakapan = list(pesan)
        galat_terakhir: Exception | None = None

        for _ in range(max(1, percobaan)):
            balasan = self.chat(percakapan, peran=peran, format_json=True, suhu=0.0)
            mentah = balasan.get("content", "")
            try:
                return skema.model_validate_json(mentah)
            except (ValidationError, ValueError) as exc:
                galat_terakhir = exc
                LOG.warning("keluaran tidak sesuai skema %s, mencoba ulang", skema.__name__)
                percakapan = percakapan + [
                    {"role": "assistant", "content": mentah},
                    {
                        "role": "user",
                        "content": (
                            "JSON di atas tidak lolos validasi. Galatnya:\n"
                            f"{exc}\n\n"
                            "Kirim ulang HANYA objek JSON yang benar, tanpa penjelasan."
                        ),
                    },
                ]

        raise GalatOllama(
            f"Model gagal menghasilkan {skema.__name__} yang valid setelah "
            f"{percobaan} percobaan. Galat terakhir: {galat_terakhir}"
        )

    # ------------------------------------------------------------- embedding
    def embed(self, teks: Sequence[str]) -> list[list[float]]:
        model = self.p.model_untuk("embedding")
        hasil = self._panggil("/api/embed", {"model": model, "input": list(teks)}, model)
        return hasil["embeddings"]

    # -------------------------------------------------------------- internal
    def _suhu(self, peran: str) -> float:
        return {
            "ekstraksi": self.p.suhu_ekstraksi,
            "agen": self.p.suhu_agen,
        }.get(peran, self.p.suhu_chat)

    def _panggil(self, jalur: str, muatan: dict, model: str) -> dict:
        try:
            r = self._sesi.post(f"{self.host}{jalur}", json=muatan, timeout=self.p.timeout)
        except requests.RequestException as exc:
            raise GalatOllama(f"Permintaan ke {self.host}{jalur} gagal: {exc}") from exc
        self._periksa(r, model)
        return r.json()

    @staticmethod
    def _periksa(r: requests.Response, model: str) -> None:
        if r.status_code == 404:
            raise ModelTidakAda(
                f"Ollama tidak mengenal model {model!r}. Tarik dengan: ollama pull {model}"
            )
        if r.status_code >= 400:
            raise GalatOllama(f"Ollama menjawab {r.status_code}: {r.text[:500]}")


_KLIEN: KlienOllama | None = None


def klien() -> KlienOllama:
    """Klien bersama. Satu sesi HTTP dipakai ulang lintas halaman Streamlit."""
    global _KLIEN
    if _KLIEN is None:
        _KLIEN = KlienOllama()
    return _KLIEN


def ringkas_anggaran() -> dict[str, Any]:
    """Ringkasan untuk ditampilkan di sidebar sebelum demo dimulai."""
    total, per_model = PENGATURAN.anggaran_terpakai()
    return {
        "profil": PENGATURAN.profil,
        "model": dict(PENGATURAN.model),
        "perkiraan_gb": per_model,
        "total_gb": round(total, 1),
        "anggaran_gb": ANGGARAN_GB,
        "muat": total <= ANGGARAN_GB,
    }
