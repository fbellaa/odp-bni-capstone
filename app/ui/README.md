# Antarmuka demo — Streamlit

Frontend live demo untuk **Agentic AI Copilot untuk Keputusan Kredit Komersial**.
Susunan halaman mengikuti proposal bagian 9.3.

> **Status: data dummy.** Model PD/LGD, lapisan graf, dan agen belum tersambung.
> Seluruh angka dibangkitkan `lib/dummy_data.py` dan dihitung `lib/mock_engine.py`
> secara deterministik (seed tetap) supaya interaksi demo tetap responsif dan
> tampilan konsisten antar sesi.

## Batas segmen yang dipakai antarmuka

Batas pada proposal 3.5 bukan keterangan tambahan — ia menjadi rentang slider,
filter tabel, dan aspek pertama pada gerbang kepatuhan.

| Dimensi | Rentang |
| --- | --- |
| Gross annual sales | Rp 30 M – 300 M |
| Maksimum kredit | Rp 10 M – 150 M |
| Average balance | Rp 10 M – 50 M |

Seluruh nominal dibaca dalam miliar rupiah (`lib/format.py: miliar()`), dan rasio
keuangan ditulis dengan satuan `x` memakai koma desimal Indonesia.

## Menjalankan

```bash
cd app/ui
pip install -r requirements.txt
streamlit run app.py
```

Untuk penempatan di belakang nginx sesuai proposal 9.2:

```bash
streamlit run app.py \
  --server.port=8501 --server.address=0.0.0.0 \
  --server.baseUrlPath=/ --server.enableCORS=false \
  --server.enableXsrfProtection=false   # hanya untuk lingkungan demo lokal
```

## Susunan berkas

```
app/ui/
├─ app.py                              Beranda + ringkasan portofolio komersial
├─ pages/
│  ├─ 1_Copilot_Pengajuan.py           Teks bebas -> jejak agen -> gerbang kepatuhan -> memo
│  ├─ 2_Simulasi_What_If.py            Slider struktur fasilitas, covenant, kurva sensitivitas
│  ├─ 3_Struktur_Grup_dan_Jaringan.py  Subgraf ego, penelusuran pemilik manfaat, anomali
│  ├─ 4_Portofolio_dan_Eksposur_Grup.py Konsentrasi grup, BMPK, uji tekanan, ambang rating
│  ├─ 5_Kesehatan_Model.py             Metrik, PSI, ablasi graf, evaluasi agen, kualitas data
│  └─ 6_Dashboard_BI.py                Iframe Metabase (+ pratinjau pengganti)
└─ lib/
   ├─ dummy_data.py                    SATU-SATUNYA sumber data dummy
   ├─ mock_engine.py                   PD/LGD/EL/pricing, covenant, dan gerbang kepatuhan
   ├─ memo.py                          Penyusunan draft credit memo komersial
   ├─ tampilan.py                      Komponen bersama (graf, BMPK, gerbang, reason code)
   └─ format.py                        Format rupiah, miliar, persen, rasio
```

## Yang berubah dari versi UMKM

- Kosakata dan seluruh angka pindah ke segmen komersial: badan hukum, grup usaha,
  pemilik manfaat, counterparty dagang, BMPK, dan rating internal.
- Fitur model mengikuti proposal bagian 6: DER, debt to EBITDA, interest coverage,
  konversi EBITDA ke kas, tingkat pemakaian plafon, saldo giro rata-rata, ditambah
  blok fitur graf (konsentrasi pembeli dan pemasok, gagal bayar 1-hop, porsi
  eksposur grup).
- **Gerbang kepatuhan** (proposal 5.3) menjadi komponen tampilan tersendiri:
  kewenangan komite, BMPK grup, agunan, covenant, afiliasi, dan batas segmen —
  masing-masing disertai pasal dan angka penyesuaian yang membuatnya patuh.
- Halaman 3 menambahkan **penelusuran kepemilikan berlapis** sampai pemilik
  manfaat akhir dan panel eksposur grup terhadap BMPK.
- Halaman 4 berganti sudut pandang dari komunitas UMKM ke konsentrasi grup
  debitur, posisi covenant per kelas rating, dan klaster ekosistem.
- Halaman 5 menambahkan tab evaluasi lapisan agen dan RAG kepatuhan.

## Menyambungkan ke FastAPI nanti

Tidak ada logika bisnis di halaman — seluruh halaman hanya memanggil fungsi di
`lib/`. Untuk beralih ke data asli, cukup ganti isi fungsi berikut dengan
pemanggilan HTTP dan pertahankan bentuk keluarannya:

| Fungsi di `lib/` | Endpoint tujuan |
| --- | --- |
| `dummy_data.daftar_pengajuan()` | `GET /api/applications` |
| `dummy_data.daftar_grup()` | `GET /api/groups` |
| `dummy_data.subgraf_ego(id, hops)` | `GET /api/entity_network` |
| `dummy_data.penelusuran_kepemilikan(id)` | `GET /api/ownership_chain` |
| `dummy_data.score_network_risk(id)` | `POST /api/score_network_risk` |
| `dummy_data.daftar_komunitas()` | `GET /api/communities` |
| `dummy_data.counterparty_penting()` | `GET /api/counterparties` |
| `dummy_data.metrik_model()`, `evaluasi_agen()` dll. | `GET /api/model_health` |
| `dummy_data.rencana_agen()` + jejak | `GET /api/agent/run` (Server-Sent Events) |
| `mock_engine.recommend_limit_pricing()` | `POST /api/score` |
| `mock_engine.check_credit_policy()` | `POST /api/check_credit_policy` |

Variabel lingkungan yang dibaca:

- `METABASE_EMBED_URL` — URL dashboard Metabase untuk halaman 6.

## Catatan teknis (proposal 9.5)

- Hasil pemanggilan agen disimpan di `st.session_state` (kunci `copilot_*`)
  supaya jejak tidak hilang saat skrip dijalankan ulang oleh interaksi lain.
- Ukuran subgraf yang dikirim ke antarmuka dibatasi (bawaan 60 simpul).
- Setiap visual graf didampingi tabel peringkat.
- Halaman 1 menyediakan **mode demo luring** untuk mengalihkan pemanggilan LLM
  ke respons yang sudah direkam.
