"""Penyusunan draft credit memo.

Sesuai prinsip pada proposal, memo hanya merangkai angka yang keluar dari tool —
tidak ada angka baru yang dikarang di lapisan ini.
"""
from __future__ import annotations

from datetime import date

from lib import mock_engine
from lib.format import persen, rupiah


def susun_memo(
    application_id: str,
    entitas: dict,
    hasil: mock_engine.HasilSkor,
    network_risk: dict,
    kebijakan,
    dokumen_kurang,
) -> str:
    keputusan = mock_engine.keputusan_dari_hasil(hasil)
    pemicu = network_risk.get("pola", [])

    baris_kontribusi = "\n".join(
        f"{i + 1}. {k.fitur} (nilai {k.nilai}) — "
        f"{'menaikkan' if k.dampak > 0 else 'menurunkan'} risiko "
        f"({k.dampak:+.3f} log-odds)"
        for i, k in enumerate(hasil.kontribusi[:3])
    )
    baris_kebijakan = "\n".join(f"- **{p['pasal']}** — {p['isi']}" for p in kebijakan)
    baris_dokumen = "\n".join(f"- [ ] {d}" for d in dokumen_kurang)
    baris_pemicu = (
        "\n".join(f"- {p['deskripsi']} ({p['bukti']})" for p in pemicu)
        if pemicu else "- Tidak ada pola anomali jaringan yang terpicu."
    )
    baris_catatan = "\n".join(f"- {c}" for c in hasil.catatan) if hasil.catatan else "- Tidak ada catatan tambahan."

    return f"""# DRAFT CREDIT MEMO
**Nomor pengajuan:** {application_id}
**Tanggal:** {date.today():%d %B %Y}
**Fasilitas:** Kredit modal kerja mikro dan kecil
**Disusun oleh:** Agentic AI Copilot (draft — wajib ditelaah analis kredit)

---

## 1. Profil pengajuan
| Item | Nilai |
| --- | --- |
| Sektor usaha | {entitas['sektor']} |
| Wilayah | {entitas['wilayah']} |
| Lama usaha | {entitas['lama_usaha_thn']:.1f} tahun |
| Omzet bulanan | {rupiah(entitas['omzet_bulanan'])} |
| Plafon diminta | {rupiah(entitas['plafon'])} |
| Tenor diminta | {entitas['tenor_bulan']} bulan |
| Jenis agunan | {entitas['jenis_agunan']} |
| Nilai taksasi agunan | {rupiah(entitas['nilai_agunan'])} |

## 2. Hasil penilaian model
| Komponen | Nilai |
| --- | --- |
| Probability of default | {persen(hasil.pd)} |
| Grade risiko | {hasil.grade} |
| Loss given default | {persen(hasil.lgd)} |
| Exposure at default | {rupiah(hasil.ead)} |
| Expected loss | {rupiah(hasil.expected_loss)} |
| Debt service coverage ratio | {hasil.dscr:.2f}x |
| Skor risiko jaringan | {network_risk['skor']:.0f} / 100 |

## 3. Faktor pendorong utama keputusan
{baris_kontribusi}

## 4. Temuan lapisan graf
{baris_pemicu}

> Skor risiko jaringan dilaporkan terpisah dan tidak dilebur ke dalam PD, agar
> alasannya tetap dapat dibaca dan diaudit.

## 5. Rujukan kebijakan kredit
{baris_kebijakan}

## 6. Usulan keputusan
**{keputusan}**

| Item | Usulan |
| --- | --- |
| Limit | {rupiah(hasil.limit_usulan)} |
| Tenor | {hasil.tenor_usulan} bulan |
| Pricing | {persen(hasil.pricing)} efektif per tahun |
| Angsuran per bulan | {rupiah(hasil.angsuran)} |

### Catatan dan syarat
{baris_catatan}

## 7. Dokumen yang masih kurang
{baris_dokumen}

---

*Dokumen ini adalah draft yang dihasilkan sistem pendukung keputusan. Keputusan
final berada pada pejabat pemutus dan dicatat pada audit log. Seluruh data yang
dipakai pada demo ini bersifat sintetis.*
"""
