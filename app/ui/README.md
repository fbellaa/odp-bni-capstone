# Antarmuka demo — Streamlit

Frontend live demo untuk **Agentic AI Copilot untuk Keputusan Kredit UMKM**.
Susunan halaman mengikuti proposal bagian 9.3.

> **Status: data dummy.** Model PD/LGD, lapisan graf, dan agen belum tersambung.
> Seluruh angka dibangkitkan `lib/dummy_data.py` dan dihitung `lib/mock_engine.py`
> secara deterministik (seed tetap) supaya interaksi demo tetap responsif dan
> tampilan konsisten antar sesi.

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
├─ app.py                          Beranda + ringkasan portofolio
├─ pages/
│  ├─ 1_Copilot_Pengajuan.py       Teks bebas -> jejak agen -> skor -> credit memo
│  ├─ 2_Simulasi_What_If.py        Slider plafon/tenor/agunan, kurva sensitivitas
│  ├─ 3_Jaringan_Entitas.py        Subgraf ego, komunitas, pola anomali
│  ├─ 4_Portofolio_dan_Komunitas.py Konsentrasi eksposur, uji tekanan, ambang skor
│  ├─ 5_Kesehatan_Model.py         Metrik, PSI, uji ablasi graf, kualitas data
│  └─ 6_Dashboard_BI.py            Iframe Metabase (+ pratinjau pengganti)
└─ lib/
   ├─ dummy_data.py                SATU-SATUNYA sumber data dummy
   ├─ mock_engine.py               Perhitungan PD/LGD/EL/pricing tiruan
   ├─ memo.py                      Penyusunan draft credit memo
   ├─ tampilan.py                  Komponen tampilan bersama (graf, reason code)
   └─ format.py                    Format rupiah dan persen
```

## Menyambungkan ke FastAPI nanti

Tidak ada logika bisnis di halaman — seluruh halaman hanya memanggil fungsi di
`lib/`. Untuk beralih ke data asli, cukup ganti isi fungsi berikut dengan
pemanggilan HTTP dan pertahankan bentuk keluarannya:

| Fungsi di `lib/` | Endpoint tujuan |
| --- | --- |
| `dummy_data.daftar_pengajuan()` | `GET /api/applications` |
| `dummy_data.subgraf_ego(id, hops)` | `GET /api/entity_network` |
| `dummy_data.score_network_risk(id)` | `POST /api/score_network_risk` |
| `dummy_data.daftar_komunitas()` | `GET /api/communities` |
| `dummy_data.counterparty_penting()` | `GET /api/counterparties` |
| `dummy_data.metrik_model()` dll. | `GET /api/model_health` |
| `dummy_data.rencana_agen()` + jejak | `GET /api/agent/run` (Server-Sent Events) |
| `mock_engine.recommend_limit_pricing()` | `POST /api/score` |

Variabel lingkungan yang dibaca:

- `METABASE_EMBED_URL` — URL dashboard Metabase untuk halaman 6.

## Catatan teknis (proposal 9.5)

- Hasil pemanggilan agen disimpan di `st.session_state` (kunci `copilot_*`)
  supaya jejak tidak hilang saat skrip dijalankan ulang oleh interaksi lain.
- Ukuran subgraf yang dikirim ke antarmuka dibatasi (bawaan 60 simpul).
- Setiap visual graf didampingi tabel peringkat.
- Halaman 1 menyediakan **mode demo luring** untuk mengalihkan pemanggilan LLM
  ke respons yang sudah direkam.
