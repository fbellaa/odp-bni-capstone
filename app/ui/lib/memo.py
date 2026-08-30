"""Penyusunan draft credit memo komersial.

Sesuai prinsip pada proposal, memo hanya merangkai angka yang keluar dari tool —
tidak ada angka baru yang dikarang di lapisan ini. Aturan yang sama berlaku
untuk pasal: bagian rujukan kebijakan diisi hasil penelusuran korpus
(`lib/kebijakan.rujukan_pengajuan`), dan ketika korpus tidak bisa ditelusuri
bagian itu menyebutkan sebabnya alih-alih menampilkan pasal pengganti. Memo
yang diunduh adalah bagian yang paling terlihat resmi; pasal karangan di
dalamnya lebih berbahaya daripada bagian yang kosong. Setiap butir kepatuhan
tetap disertai pasal yang menjadi dasarnya.
"""
from __future__ import annotations

from datetime import date

from lib import kebijakan as kb, mock_engine
from lib.format import kali, persen, rupiah


def _teks_jaringan(network_risk: dict) -> str:
    """Nilai indikator jaringan, atau sebab kenapa ia tidak ada.

    Memo tidak boleh menuliskan angka ketika pencocokan afiliasi tidak bisa
    dijalankan: sel kosong yang jujur lebih berguna bagi komite daripada nol
    yang terbaca sebagai "tidak ada risiko".
    """
    skor = network_risk.get("skor")
    if skor is None:
        sebab = "; ".join(network_risk.get("catatan") or []) or "tidak dapat dihitung"
        return f"tidak dapat dihitung — {sebab}"
    return f"{skor:.0f} / 100 (indikator beraturan, bukan model terlatih)"


def _bagian_ews(pantauan) -> str:
    """Status peringatan dini afiliasi, atau sebab kenapa ia tidak ada.

    Yang dilaporkan bukan skor pemohon — model EWS menilai fasilitas berjalan,
    dan pemohon belum punya satu pun. Yang dilaporkan adalah kondisi debitur
    eksisting yang terhubung dengannya.
    """
    if pantauan is None:
        return ("- Peringatan dini afiliasi tidak dijalankan: artefak EWS atau panel "
                "bulanan tidak tersedia.")
    if getattr(pantauan, "tabel", None) is None or pantauan.tabel.empty:
        return (f"- Tidak ada fasilitas afiliasi pada panel bulanan sampai "
                f"{pantauan.tanggal:%B %Y}. Ini batas penelaahan, bukan temuan bersih.")
    cacah = pantauan.cacah_pita()
    return (
        f"- {len(pantauan.tabel)} fasilitas afiliasi terpantau pada snapshot terakhir "
        f"sebelum {pantauan.tanggal:%B %Y}: "
        f"{cacah.get('HIGH', 0)} peringatan dini, {cacah.get('MEDIUM', 0)} perlu "
        f"diperhatikan, {cacah.get('LOW', 0)} pantauan biasa.\n"
        f"- {pantauan.jumlah_alarm} di antaranya melewati ambang alarm "
        f"{pantauan.ambang:.4f} (disetel pada recall 80% populasi latih)."
    )


def _bagian_kebijakan(kebijakan, catatan) -> str:
    """Daftar pasal hasil penelusuran korpus, atau sebab kenapa ia kosong.

    Sel kosong yang jujur berlaku di sini persis seperti pada indikator
    jaringan: komite yang membaca "korpus belum ditelusuri" tahu bagian ini
    masih pekerjaannya, sedangkan komite yang membaca empat pasal karangan
    tidak tahu apa-apa dan mengira sudah tahu.
    """
    catatan = list(catatan or [])
    baris = []
    for p in kebijakan or []:
        jejak = kb.jejak_sumber(p)
        topik = ", ".join(p.get("topik") or []) or "penelusuran umum"
        baris.append(
            # Dua spasi di ujung baris: satu butir daftar dengan tiga baris,
            # bukan tiga baris yang dilebur markdown menjadi satu paragraf.
            f"- **{p['pasal']}** — {jejak}  \n"
            f"  Ditelusuri untuk: {topik}  \n"
            f"  > {p['isi']}"
        )
    if not baris:
        baris.append(
            "> **Tidak ada rujukan yang dapat dikutip.** Korpus kebijakan tidak "
            "ditelusuri pada penyusunan memo ini, sehingga bagian ini sengaja "
            "dibiarkan kosong. Analis wajib mengisinya dari peraturan yang berlaku "
            "sebelum memo dibawa ke komite — sistem tidak menyusun pasal sendiri."
        )
    if catatan:
        baris.append("")
        baris.append("**Batas penelusuran**")
        baris.extend(f"- {c}" for c in catatan)
    return "\n".join(baris)


def _bagian_kontribusi(kontribusi, sumber: str | None) -> str:
    """Faktor pendorong keputusan, beserta model yang menghitungnya.

    Sampai perbaikan ini, bagian ini diisi `HasilSkor.kontribusi` — keluaran
    logistik tiruan `mock_engine.score_pd()`. Angka PD-nya sendiri sudah lama
    ditimpa model sungguhan, tetapi daftar pendorongnya tidak: memo yang diunduh
    mencantumkan tiga faktor lengkap dengan log-odds yang tidak pernah dihitung
    model mana pun, sementara layar menampilkan SHAP yang benar. Persis pola
    yang sama dengan pasal karangan pada bagian rujukan.
    """
    baris = [
        f"{i + 1}. {k.fitur} (nilai {k.nilai}) — "
        f"{'menaikkan' if k.dampak > 0 else 'menurunkan'} risiko "
        f"({k.dampak:+.3f} log-odds)"
        for i, k in enumerate((kontribusi or [])[:3])
    ]
    if not baris:
        return (
            "> **Tidak ada faktor pendorong yang dapat dilaporkan.** Model PD tidak "
            "tersedia pada penyusunan memo ini, sehingga tidak ada nilai SHAP yang "
            "bisa dikutip. Bagian ini sengaja dibiarkan kosong daripada diisi "
            "pendorong dari mesin demo."
        )
    if sumber:
        baris.append("")
        baris.append(f"*Sumber: {sumber}.*")
    return "\n".join(baris)


def susun_memo(
    application_id: str,
    entitas: dict,
    hasil: mock_engine.HasilSkor,
    network_risk: dict,
    kebijakan,
    dokumen_kurang,
    gerbang=None,
    catatan_kebijakan=None,
    pantauan_ews=None,
    kontribusi=None,
    sumber_kontribusi=None,
) -> str:
    gerbang = gerbang or []
    keputusan = mock_engine.keputusan_dari_hasil(hasil, gerbang or None)
    pemicu = network_risk.get("pola", [])
    cov = hasil.covenant

    baris_kontribusi = _bagian_kontribusi(kontribusi, sumber_kontribusi)
    baris_kebijakan = _bagian_kebijakan(kebijakan, catatan_kebijakan)
    baris_ews = _bagian_ews(pantauan_ews)
    baris_dokumen = "\n".join(f"- [ ] {d}" for d in dokumen_kurang)
    baris_pemicu = (
        "\n".join(f"- {p['deskripsi']} ({p['bukti']})" for p in pemicu)
        if pemicu else "- Tidak ada pola anomali struktur yang terpicu."
    )
    baris_catatan = "\n".join(f"- {c}" for c in hasil.catatan) if hasil.catatan else "- Tidak ada catatan tambahan."
    # Aturan internal dan rujukan korpus dipisah menjadi dua kolom. Digabung,
    # aturan internal terbaca seolah punya dasar pasal — itu yang dulu terjadi
    # ketika kolomnya diisi nomor karangan.
    baris_gerbang = "\n".join(
        f"| {a['aspek']} | {a['status']} | {a['temuan']} | {a.get('aturan', '-')} | "
        f"{kb.label_dasar(a)} | {a['tindakan']} |"
        for a in gerbang
    ) or "| - | - | Gerbang kepatuhan belum dijalankan | - | - | - |"

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
| Skor default 12 bulan | {persen(hasil.pd)} |
| Pita risiko | {hasil.grade} |
| Loss given default | {persen(hasil.lgd)} |
| Exposure at default | {rupiah(hasil.ead, singkat=True)} |
| Expected loss (indikatif) | {rupiah(hasil.expected_loss, singkat=True)} |
| Interest coverage ratio | {kali(hasil.icr)} |
| Debt to EBITDA | {kali(hasil.debt_to_ebitda)} |
| Debt service coverage ratio | {kali(hasil.dscr)} |
| Pertanggungan agunan | {persen(hasil.coverage_agunan, 0)} |
| Indikator risiko jaringan | {_teks_jaringan(network_risk)} |

## 3. Faktor pendorong utama keputusan
{baris_kontribusi}

## 4. Temuan lapisan graf dan struktur grup
{baris_pemicu}

**Peringatan dini afiliasi**
{baris_ews}

| Item | Nilai |
| --- | --- |
| Eksposur grup berjalan | {rupiah(hasil.eksposur_grup, singkat=True)} |
| Sisa ruang BMPK grup | {rupiah(hasil.ruang_bmpk, singkat=True)} |
| Batas maksimum pemberian kredit grup | {rupiah(hasil.batas_bmpk, singkat=True)} |
| Asal angka eksposur grup | {hasil.sumber_bmpk} |

> Skor risiko jaringan dilaporkan terpisah dan tidak dilebur ke dalam PD, agar
> alasannya tetap dapat dibaca dan diaudit komite kredit.

## 5. Gerbang kepatuhan
| Aspek | Status | Temuan | Aturan yang diuji | Dasar pada korpus | Tindakan |
| --- | --- | --- | --- | --- | --- |
{baris_gerbang}

Status kepatuhan keseluruhan: **{mock_engine.status_kepatuhan(gerbang) if gerbang else '-'}**

## 6. Rujukan kebijakan
*Dikutip dari korpus `docs/policies` yang terindeks. Bagian ini tidak pernah diisi sistem tanpa berkas sumber.*

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

### Covenant wajib pita {hasil.grade.lower()}
- Debt to equity ratio maksimum {kali(cov['der_maks'])}
- Interest coverage ratio minimum {kali(cov['icr_min'])}
{('- Debt to EBITDA maksimum ' + kali(cov['debt_to_ebitda_maks'])) if cov.get('debt_to_ebitda_maks') else ''}
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
