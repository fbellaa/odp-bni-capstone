"""Penyusunan draft credit memo komersial.

Sesuai prinsip pada proposal, memo hanya merangkai angka yang keluar dari tool —
tidak ada angka baru yang dikarang di lapisan ini. Setiap butir kepatuhan
disertai pasal yang menjadi dasarnya.
"""
from __future__ import annotations

from datetime import date

from lib import mock_engine
from lib.format import kali, persen, rupiah


def susun_memo(
    application_id: str,
    entitas: dict,
    hasil: mock_engine.HasilSkor,
    network_risk: dict,
    kebijakan,
    dokumen_kurang,
    gerbang=None,
) -> str:
    gerbang = gerbang or []
    keputusan = mock_engine.keputusan_dari_hasil(hasil, gerbang or None)
    pemicu = network_risk.get("pola", [])
    cov = hasil.covenant

    baris_kontribusi = "\n".join(
        f"{i + 1}. {k.fitur} (nilai {k.nilai}) — "
        f"{'menaikkan' if k.dampak > 0 else 'menurunkan'} risiko "
        f"({k.dampak:+.3f} log-odds)"
        for i, k in enumerate(hasil.kontribusi[:3])
    )
    baris_kebijakan = "\n".join(
        f"- **{p['pasal']}** ({p.get('versi', 'versi berlaku')}) — {p['isi']}" for p in kebijakan
    )
    baris_dokumen = "\n".join(f"- [ ] {d}" for d in dokumen_kurang)
    baris_pemicu = (
        "\n".join(f"- {p['deskripsi']} ({p['bukti']})" for p in pemicu)
        if pemicu else "- Tidak ada pola anomali struktur yang terpicu."
    )
    baris_catatan = "\n".join(f"- {c}" for c in hasil.catatan) if hasil.catatan else "- Tidak ada catatan tambahan."
    baris_gerbang = "\n".join(
        f"| {a['aspek']} | {a['status']} | {a['temuan']} | `{a['pasal']}` | {a['tindakan']} |"
        for a in gerbang
    ) or "| - | - | Gerbang kepatuhan belum dijalankan | - | - |"

    return f"""# DRAFT CREDIT MEMO — SEGMEN KOMERSIAL
**Nomor pengajuan:** {application_id}
**Tanggal:** {date.today():%d %B %Y}
**Debitur:** {entitas.get('nama_debitur', '-')}
**Grup usaha:** {entitas.get('jumlah_entitas_grup', 1)} entitas tergabung sebagai satu grup debitur
**Fasilitas:** Kredit modal kerja dan investasi komersial
**Kewenangan pemutus:** {hasil.komite_pemutus}
**Disusun oleh:** Agentic AI Copilot (draft — wajib ditelaah analis kredit)

---

## 1. Profil pengajuan
| Item | Nilai |
| --- | --- |
| Sektor usaha | {entitas['sektor']} |
| Wilayah | {entitas['wilayah']} |
| Umur badan usaha | {entitas['umur_usaha_thn']:.0f} tahun |
| Penjualan tahunan | {rupiah(entitas['penjualan_tahunan'], singkat=True)} |
| EBITDA margin | {persen(entitas['ebitda_margin'])} |
| Debt to equity ratio | {kali(entitas['der'])} |
| Saldo giro rata-rata | {rupiah(entitas.get('saldo_giro_rata', 0), singkat=True)} |
| Plafon diminta | {rupiah(entitas['plafon'], singkat=True)} |
| Tenor diminta | {entitas['tenor_bulan']} bulan |
| Jenis agunan | {entitas['jenis_agunan']} |
| Nilai taksasi agunan | {rupiah(entitas['nilai_agunan'], singkat=True)} |

## 2. Hasil penilaian model
| Komponen | Nilai |
| --- | --- |
| Probability of default | {persen(hasil.pd)} |
| Rating internal | {hasil.grade} |
| Loss given default | {persen(hasil.lgd)} |
| Exposure at default | {rupiah(hasil.ead, singkat=True)} |
| Expected loss | {rupiah(hasil.expected_loss, singkat=True)} |
| Interest coverage ratio | {kali(hasil.icr)} |
| Debt to EBITDA | {kali(hasil.debt_to_ebitda)} |
| Debt service coverage ratio | {kali(hasil.dscr)} |
| Pertanggungan agunan | {persen(hasil.coverage_agunan, 0)} |
| Skor risiko jaringan | {network_risk['skor']:.0f} / 100 |

## 3. Faktor pendorong utama keputusan
{baris_kontribusi}

## 4. Temuan lapisan graf dan struktur grup
{baris_pemicu}

| Item | Nilai |
| --- | --- |
| Eksposur grup berjalan | {rupiah(hasil.eksposur_grup, singkat=True)} |
| Sisa ruang BMPK grup | {rupiah(hasil.ruang_bmpk, singkat=True)} |
| Batas maksimum pemberian kredit grup | {rupiah(mock_engine.BATAS_BMPK_GRUP, singkat=True)} |

> Skor risiko jaringan dilaporkan terpisah dan tidak dilebur ke dalam PD, agar
> alasannya tetap dapat dibaca dan diaudit komite kredit.

## 5. Gerbang kepatuhan
| Aspek | Status | Temuan | Pasal | Tindakan |
| --- | --- | --- | --- | --- |
{baris_gerbang}

Status kepatuhan keseluruhan: **{mock_engine.status_kepatuhan(gerbang) if gerbang else '-'}**

## 6. Rujukan kebijakan kredit komersial
{baris_kebijakan}

## 7. Usulan keputusan
**{keputusan}**

| Item | Usulan |
| --- | --- |
| Limit | {rupiah(hasil.limit_usulan, singkat=True)} |
| Tenor | {hasil.tenor_usulan} bulan |
| Pricing | {persen(hasil.pricing)} efektif per tahun |
| Angsuran per bulan | {rupiah(hasil.angsuran, singkat=True)} |
| Kewenangan pemutus | {hasil.komite_pemutus} |

### Covenant wajib kelas rating {hasil.grade}
- Debt to equity ratio maksimum {kali(cov['der_maks'])}
- Interest coverage ratio minimum {kali(cov['icr_min'])}
- Debt service coverage ratio minimum {kali(cov['dscr_min'])}
- Frekuensi pengujian: {cov['uji'].lower()}
- Larangan penjaminan silang baru tanpa persetujuan tertulis bank

### Catatan dan syarat
{baris_catatan}

## 8. Dokumen yang masih kurang
{baris_dokumen}

---

*Dokumen ini adalah draft yang dihasilkan sistem pendukung keputusan. Keputusan
final berada pada komite kredit sesuai matriks kewenangan dan dicatat pada audit
log. Seluruh data yang dipakai pada demo ini bersifat sintetis.*
"""
