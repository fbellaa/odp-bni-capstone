# Basis kebijakan (untuk RAG)

Dokumen regulasi resmi yang jadi sumber pengetahuan copilot pengajuan kredit
komersial. Teks peraturan perundang-undangan Indonesia (termasuk POJK/SEOJK)
dikecualikan dari hak cipta (UU 28/2014 Pasal 42), sehingga teks lengkapnya
boleh disimpan di repo ini.

| Berkas | Nomor | Judul | Berlaku | Sumber |
|---|---|---|---|---|
| [pojk-40-pojk.03-2019.pdf](pojk-40-pojk.03-2019.pdf) | 40/POJK.03/2019 | Penilaian Kualitas Aset Bank Umum | 19 Des 2019 (LN 2019 No. 247, TLN No. 6440) | [peraturan.go.id](https://peraturan.go.id/id/peraturan-ojk-no-40-pojk-03-2019-tahun-2019) |

Cakupan dokumen ini: penilaian & penetapan kualitas aset produktif dan
non-produktif, cadangan penurunan nilai (CKPN), restrukturisasi kredit,
hapus buku, dan pelaporan berkala kolektibilitas.

## Belum termasuk (di luar scope saat ini)

- POJK 2/POJK.03/2022 — versi syariah, relevan jika portofolio komersial
  mencakup pembiayaan syariah.
- Regulasi BMPK / konsentrasi kredit.
- Regulasi tata kelola risiko kredit umum.

## Catatan untuk pipeline RAG

Ini dokumen sumber mentah (PDF asli), belum di-chunk atau di-embed. Saat
membangun index RAG, pertahankan referensi nomor pasal per chunk agar
jawaban copilot bisa disitasi balik ke sumbernya.
