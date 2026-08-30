# Antarmuka demo — Streamlit

Frontend live demo untuk **Agentic AI Copilot untuk Keputusan Kredit Komersial**.
Susunan halaman mengikuti proposal bagian 9.3.

> **Status: model sungguhan, sebagian lapisan masih demo.** PD, EWS, LGD, dan
> ruang klaster portofolio dibaca dari `ml/artifacts` di atas `data/gold` lewat
> `lib/model_nyata.py`. Risiko jaringan sejak kini dihitung `lib/risiko_jaringan.py`
> dari `feat_graf_pit`, `fact_afiliasi_tersembunyi`, dan `fact_agunan` — komponennya
> nyata, pembobotannya keputusan kebijakan, jadi ia disebut **indikator** dan bukan
> model. Yang masih dilayani `lib/dummy_data.py` dan `lib/mock_engine.py`: rantai
> limit/pricing/covenant, pratinjau Dashboard BI selama Metabase belum jalan, dan
> rencana tool pada jalur tanpa agen. Fitur perilaku dan graf yang tidak tertulis
> di berkas kini memakai median portofolio, bukan konstanta tulisan tangan, dan
> tab Reason code menampilkan asal tiap fitur satu per satu. Reason code pada memo tidak lagi termasuk:
> ia memakai nilai SHAP model PD, dan dikosongkan beserta sebabnya bila artefak
> PD tidak ada. Rumus rantai itu tetap deterministik, tetapi
> parameternya tidak lagi ditulis tangan: `lib/parameter_kebijakan.py` menurunkan
> recovery agunan, ambang covenant, matriks kewenangan, dan eksposur BMPK grup
> dari lapisan emas, dan tiap angka membawa keterangan asalnya. Rujukan kebijakan pada memo tidak lagi
> termasuk: ia dikutip dari korpus terindeks lewat `lib/kebijakan.py`, dan
> dikosongkan beserta sebabnya bila korpus tidak bisa ditelusuri. Halaman selalu
> menyebutkan bagian mana yang demo.

## Palet

| Warna | Kode | Tugas |
| --- | --- | --- |
| Jingga | `#FF8000` | tindakan (tombol utama), peringatan, sisi buruk sebuah ukuran |
| Tosca | `#40C0C0` | struktur: kepala halaman, batang grafik, sisi baik sebuah ukuran |
| Abu | `#808080` | teks sekunder, garis, kategori netral |
| Putih | `#FFFFFF` | permukaan kartu dan tabel |

Tingkat risiko selalu dibaca sebagai satu tanjakan tosca → jingga → jingga
gelap, bukan hijau-kuning-merah, sehingga urutannya tetap terbaca saat dicetak
hitam putih atau dilihat mata yang sulit membedakan merah dan hijau. Seluruh
nilainya didefinisikan di `lib/tampilan.py`; tidak ada halaman yang menuliskan
kode warna sendiri.

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
pip install -r app/ui/requirements.txt
streamlit run app/ui/Copilot_Pengajuan.py
```

Untuk penempatan di belakang nginx sesuai proposal 9.2:

```bash
streamlit run app/ui/Copilot_Pengajuan.py \
  --server.port=8501 --server.address=0.0.0.0 \
  --server.baseUrlPath=/ --server.enableCORS=false \
  --server.enableXsrfProtection=false   # hanya untuk lingkungan demo lokal
```

## Susunan berkas

```
app/ui/
├─ Copilot_Pengajuan.py                Skrip entry: unggahan PDF -> tool -> PD/LGD/klaster -> memo
├─ pages/
│  ├─ 1_Simulasi_What_If.py            Sandbox slider, covenant, kurva sensitivitas
│  ├─ 2_Struktur_Grup_dan_Jaringan.py  Subgraf ego, penelusuran pemilik manfaat, anomali
│  └─ 3_Dashboard_BI.py                Iframe Metabase (+ pratinjau pengganti)
└─ lib/
   ├─ model_nyata.py                   Artefak ml/artifacts + data gold: PD, EWS, LGD, klaster
   ├─ pipeline_copilot.py              PDF -> fakta -> entitas gabungan (jalur LLM / pola)
   ├─ copilot_lokal.py                 Satu-satunya pintu ke paket `copilot`
   ├─ risiko_jaringan.py               Indikator risiko jaringan atas data graf nyata
   ├─ kebijakan.py                     Rujukan pasal dari korpus terindeks + ceklis dokumen
   ├─ parameter_kebijakan.py           Parameter kebijakan diturunkan dari data/gold
   ├─ dummy_data.py                    Lapisan demo yang belum punya model (rencana agen)
   ├─ mock_engine.py                   Rantai limit/pricing/covenant dan gerbang kepatuhan
   ├─ memo.py                          Penyusunan draft credit memo komersial
   ├─ tampilan.py                      Tema, kartu, meter PD, peta klaster, graf
   └─ format.py                        Format rupiah, miliar, persen, rasio
```

Halaman lama `4_Portofolio_dan_Eksposur_Grup.py` dihapus, dan `7_Copilot_Lokal.py`
dilebur ke halaman 1 — satu halaman pengajuan, bukan dua yang saling menyalin.

## Yang berubah dari versi UMKM

- Kosakata dan seluruh angka pindah ke segmen komersial: badan hukum, grup usaha,
  pemilik manfaat, counterparty dagang, BMPK, dan rating internal.
- Fitur model mengikuti proposal bagian 6: DER, debt to EBITDA, interest coverage,
  konversi EBITDA ke kas, tingkat pemakaian plafon, saldo giro rata-rata, ditambah
  blok fitur graf (konsentrasi pembeli dan pemasok, gagal bayar 1-hop, porsi
  eksposur grup).
- **Gerbang kepatuhan** (proposal 5.3) menjadi komponen tampilan tersendiri:
  kewenangan komite, BMPK grup, agunan, covenant, afiliasi, dan batas segmen —
  masing-masing disertai aturan yang diuji dan angka penyesuaian yang membuatnya
  patuh. Dasar hukumnya dipetakan tangan ke pasal pada korpus (`lib/kebijakan.py:
  RUJUKAN_GERBANG`), bukan dicari lewat kemiripan: aspek yang diatur peraturan
  di luar korpus — BMPK pada POJK 32/2018, matriks kewenangan internal —
  mengatakannya, dan tidak diisi nomor pasal yang sekadar terdengar cocok.
- Halaman struktur grup menambahkan **penelusuran kepemilikan berlapis** sampai
  pemilik manfaat akhir dan panel eksposur grup terhadap BMPK.

## Parameter kebijakan yang sudah turun dari data

Rantai limit, pricing, dan gerbang kepatuhan tetap perhitungan aturan — batas
kredit punya akibat hukum, jadi ia tidak boleh keluar dari regresi. Yang dulu
salah bukan bentuk rumusnya melainkan angka yang masuk ke dalamnya. Sekarang:

| Parameter | Sumber | Catatan |
| --- | --- | --- |
| Eksposur dan batas BMPK grup | `fact_eksposur_grup` lewat `dim_debitur.grup_id` | Hanya bila cocokan afiliasi menunjuk satu grup; tanpa itu layar menyebut "tidak terukur" |
| Recovery per jenis agunan | `fact_agunan.haircut` | Deposito 1,00 · tanah 0,70 · mesin 0,50 · piutang 0,45 · persediaan 0,40 |
| Ambang covenant per pita | `fact_covenant` x pita model PD | DER, ICR, debt to EBITDA; DSCR tetap kebijakan internal |
| Suku bunga dasar per pita | `fact_pengajuan.pricing_bps` | 726 · 801 · 872 · 906 bps dari pita rendah ke sangat tinggi |
| Pagu limit per pita | `abt_pd.app_plafon_rp` persentil 95 | 89 · 91 · 79 · 71 miliar; langit-langit praktik, bukan kewenangan resmi |
| Matriks kewenangan komite | `fact_pengajuan.komite_level` | Kepala Cabang sampai Rp 25 M · Wilayah sampai Rp 75 M · Pusat di atasnya |
| Konversi EBITDA ke kas | `abt_pd.fin_cfo_to_ebitda` median 0,738 | Dihitung dari berkas pemohon bila laporan memuat arus kas operasi |
| Tingkat pemakaian plafon | `fact_fasilitas.pemakaian_plafon_pct` median 0,740 | |
| Porsi utang berbunga | `abt_pd` debt/EBITDA x EBITDA / liabilitas, median 0,878 | Menggantikan taksiran 0,50 yang membuat ICR dan DSCR terbaca lebih sehat |
| Utang berbunga eksisting | `fact_fasilitas.outstanding_rp` | Nyata begitu afiliasi tercocok; tanpa itu porsi di atas dikalikan liabilitas dokumen |
| Konsentrasi pembeli/pemasok, default 1-hop | `feat_graf_pit` median | Nyata untuk afiliasi tercocok |
| Pertanggungan agunan minimum | kebijakan internal | Tidak diturunkan dari data: lapisan emas hanya memuat coverage yang menyertai fasilitas (median 1,33, rata di semua pita), bukan ambang minimum |

Selisihnya nyata: batas BMPK bawaan Rp 750 M sementara data memakai Rp 3 T per
grup, dan arah covenant DER pada konstanta lama terbalik dari yang dipakai
portofolio. Sidebar halaman copilot menampilkan asal tiap parameter, dan angka
yang masih memakai cadangan disebut sebagai asumsi.

## Pita risiko menggantikan rating huruf

Kelas rating AAA-CCC tidak lagi dipakai antarmuka. Huruf itu dulu diturunkan
dari PD lewat ambang yang ditulis `mock_engine` sendiri — tidak ada model yang
mengeluarkannya. Penggantinya adalah pita risiko pada
`ml/artifacts/pd/pd_decision_policy.json`, yaitu kontrak artefak PD:

| Pita | Jumlah pada ABT | Default aktual | Pricing historis | DER maks | ICR min |
| --- | --- | --- | --- | --- | --- |
| Risiko rendah | 2.040 | 0,25% | 726 bps | 2,00x | 3,00x |
| Risiko sedang | 1.362 | 1,06% | 801 bps | 2,50x | 2,50x |
| Risiko tinggi | 639 | 7,28% | 872 bps | 3,00x | 2,00x |
| Risiko sangat tinggi | 142 | 47,10% | 906 bps | 3,50x | 1,50x |

Empat pita, bukan tiga: melebur dua pita teratas akan menyatukan 142 pengajuan
bertingkat default 47% dengan 639 pengajuan bertingkat default 7%.

Ambang pita diambil dari berkas kebijakan, **bukan** dari `risk_cutoffs` yang
masih tersimpan di dalam joblib PD — yang terakhir itu tertinggal dari versi
sebelumnya, dan memakainya membuat pita terendah kosong sama sekali.

Dua akibat lain dari artefak PD versi ini, yang tidak terkalibrasi:

- Layar tidak lagi menulis "PD terkalibrasi". Angkanya skor peringkat, dan yang
  dibaca komite adalah pitanya.
- **Expected loss tidak lagi masuk ke pricing.** Skor tak terkalibrasi dikali
  LGD dan EAD tidak menghasilkan rupiah yang bisa dipertanggungjawabkan, jadi
  pricing datang murni dari grid pita. Expected loss tetap dilaporkan, ditandai
  indikatif.

## EWS dipakai untuk afiliasi, bukan untuk pemohon

Model peringatan dini menilai fasilitas yang **sudah berjalan**: 26 fiturnya
hampir seluruhnya perilaku — tunggakan, kolektibilitas, pemakaian plafon,
pelanggaran covenant, beserta perubahan satu dan tiga bulannya. Pemohon baru
tidak punya satu pun dari itu, jadi EWS tidak bisa dan tidak boleh dipakai
menilai pengajuan.

Yang punya perilaku adalah afiliasinya. Debitur eksisting yang tercocok dari
dokumen pemohon diskor pada snapshot terakhir sebelum tanggal telaah, dan
hasilnya muncul pada tab **Risiko jaringan** serta bagian 4 draft credit memo:
berapa fasilitas afiliasi yang berstatus peringatan dini, dan berapa yang
melewati ambang alarm (0,0357, disetel pada recall 80%).

Pertanyaan yang dijawabnya bukan "apakah pemohon ini akan gagal bayar",
melainkan "apakah grup tempat ia berada sedang menuju masalah selagi ia meminta
fasilitas baru" — penularan dalam satu grup, yang melengkapi indikator jaringan
karena indikator itu hanya menghitung gagal bayar yang sudah terjadi.

Tiga pita EWS (LOW/MEDIUM/HIGH) sengaja tidak diseragamkan dengan empat pita
PD: ambangnya datang dari sebaran skor yang berbeda, atas target yang berbeda
pula (6 bulan pada fasilitas berjalan, bukan 12 bulan pada pengajuan).

## Menyambungkan ke FastAPI nanti

Tidak ada logika bisnis di halaman — seluruh halaman hanya memanggil fungsi di
`lib/`. Untuk beralih ke data asli, cukup ganti isi fungsi berikut dengan
pemanggilan HTTP dan pertahankan bentuk keluarannya:

| Fungsi di `lib/` | Endpoint tujuan |
| --- | --- |
| `dummy_data.daftar_pengajuan()`, `daftar_grup()` | `GET /api/applications`, `GET /api/groups` (pratinjau Dashboard BI) |
| `risiko_jaringan.skor_jaringan()` | `POST /api/score_network_risk` |
| `kebijakan.rujukan_pengajuan()` | `GET /api/policy_search` |
| `parameter_kebijakan.eksposur_grup()` | `GET /api/group_exposure` |
| `dummy_data.rencana_agen()` + jejak | `GET /api/agent/run` (Server-Sent Events) |
| `mock_engine.recommend_limit_pricing()` | `POST /api/score` |
| `mock_engine.check_credit_policy()` | `POST /api/check_credit_policy` |

Variabel lingkungan yang dibaca:

- `METABASE_EMBED_URL` — URL dashboard Metabase untuk halaman Dashboard BI.

## Catatan teknis (proposal 9.5)

- Hasil pemanggilan agen disimpan di `st.session_state` (kunci `copilot_*`)
  supaya jejak tidak hilang saat skrip dijalankan ulang oleh interaksi lain.
- Ukuran subgraf yang dikirim ke antarmuka dibatasi (bawaan 60 simpul).
- Setiap visual graf didampingi tabel peringkat.
- Halaman 1 menyediakan **mode demo luring** untuk mengalihkan pemanggilan LLM
  ke respons yang sudah direkam.
