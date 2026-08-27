# Silsilah Data: Nyata vs Sintesis

Setiap klaim "nyata" harus bisa ditelusuri ke berkas sumbernya, dan setiap angka
sintesis harus ditandai. Tanpa dokumen ini, gabungan tujuh dataset di bawah menjadi klaim
yang menyesatkan.

---

## 1. Peringatan utama

**Join antar dataset publik ini tidak ada di dunia nyata.** Baris rasio Taiwan
tidak punya hubungan apa pun dengan akun di dataset AML IBM, maupun dengan simpul
ICIJ. Yang dilakukan pipeline adalah *menempelkan* mereka pada satu `cif`
sintetis lewat `pipelines/transform/joins.py`.

Yang nyata: **sinyalnya** (label gagal bayar, rasio keuangan, tingkat pemulihan)
dan **topologinya** (derajat, siklus, kepemilikan berlapis).
Yang sintesis: **keterkaitan antar sumber**, dan seluruh konteks Indonesia.

---

## 2. Tabel status per objek

| Objek | Sumber | Status |
|---|---|---|
| Rasio keuangan & label gagal bayar | `data.csv` + `american_bankruptcy.csv` | **NYATA** |
| Tren rasio 3 tahun | `american_bankruptcy.csv` (panel) | **NYATA** |
| Distribusi rating & rasio per sektor | `corporate_rating.csv` | **NYATA** |
| Tingkat pemulihan / LGD | `SBAnational.csv` | **NYATA** |
| Tenor, revolving, kelengkapan dokumen | `SBAnational.csv` | **NYATA** |
| Topologi transfer, siklus, fan-in/out | `LI-Small_Trans.csv` | **NYATA** (sintetis di sumber, bukan buatan kita) |
| Kanal pembayaran transfer (`format_pembayaran`) | pemetaan ke sistem pembayaran Indonesia | SINTESIS - lihat 3.7 |
| Kepemilikan berlapis, rangkap jabatan, alamat dibagi | ICIJ Offshore Leaks | **NYATA** |
| Nama PT, CIF, NPWP, alamat Indonesia | Faker + aturan | SINTESIS |
| Sektor KBLI | pemetaan NAICS 2 digit | SINTESIS (turunan) |
| Skala rupiah | penskalaan peringkat | SINTESIS |
| Pengajuan, keputusan komite, pricing | aturan distribusi | SINTESIS |
| Agunan, covenant, kolektibilitas bulanan | aturan distribusi | SINTESIS |
| `grup_id` | klaster hub ICIJ | SINTESIS (turunan) |
| Nama pihak (`DIM_PIHAK.nama`) | Faker | SINTESIS - nama asli ICIJ **tidak** dibawa |
| Struktur berbagi alamat (siapa sekantor dengan siapa) | ICIJ Offshore Leaks | **NYATA** |
| Teks alamat (`DIM_ALAMAT.alamat_teks`) | Faker | SINTESIS - alamat asli ICIJ **tidak** dibawa |
| Join antar seluruh dataset | pemetaan `cif` | **SINTESIS - wajib didokumentasikan** |

---

## 3. Asumsi yang menempel pada angka "nyata"

Beberapa angka nyata butuh satu asumsi supaya bisa dihitung. Asumsi ini harus
ikut dilaporkan setiap kali angkanya dikutip.

### 3.1 Beban bunga dan ICR (`silver.build_silver_us_panel`)

`american_bankruptcy.csv` (X1..X18) **tidak memuat beban bunga**. Beban bunga
diturunkan dari identitas laba:

```
NI = (EBIT - bunga) x (1 - tarif pajak)   ->   bunga = EBIT - NI / (1 - 0,22)
bunga = max(bunga, 1% x total liabilitas)   # lantai supaya ICR tidak meledak
ICR   = EBIT / bunga
```

Tarif pajak 22% adalah asumsi tunggal, dipakai konsisten
(`silver.TARIF_PAJAK`). Semua rasio lain
(DER, debt/EBITDA, ROA, current ratio, DSO, DIO) dihitung langsung dari pos
akuntansi tanpa asumsi tambahan.

### 3.2 Siklus modal kerja

`siklus_modal_kerja_hari = DSO + DIO`. **DPO tidak dihitung** karena utang usaha
tidak tersedia di X1..X18. Angkanya karena itu lebih tinggi dari siklus kas
sebenarnya, dan tidak boleh dibandingkan langsung dengan benchmark industri.

### 3.3 Arus kas operasi

`cfo_proxy = laba bersih + penyusutan`. Laporan arus kas tidak ada di sumber.

### 3.4 Waktu pada dataset AML

Timestamp asli `LI-Small_Trans.csv` hanya mencakup **1-17 September 2022**
(6.924.049 baris). Urutan dan jarak relatif antar transfer dipertahankan, lalu
diregangkan linier ke jendela 2023-01-01 s.d. 2025-12-31 supaya snapshot bulanan
graf punya isi. Topologi tidak diubah sama sekali; hanya sumbu waktunya.

### 3.5 Subgraf AML

Dari 705.903 rekening unik, dipertahankan **20.000 rekening berderajat
tertinggi** beserta subgraf terinduksinya (297.553 transfer). Alasannya ukuran
berkas; konsekuensinya distribusi derajat tergeser ke atas dibanding populasi
penuh, dan ini harus disebut saat melaporkan metrik graf.

### 3.6 Kurs

Nilai transfer AML dikonversi ke rupiah dengan tabel kurs statis
(`silver.KURS_KE_USD`, USD/IDR 17.700 - `silver.KURS_USD_IDR`). Angka ini
hanya dipakai sebagai **bobot edge**, bukan sebagai nilai transaksi yang
berarti.

### 3.7 Kanal pembayaran (`format_pembayaran`)

Nilai asli `LI-Small_Trans.csv` dipetakan ke padanan sistem pembayaran Indonesia
di `silver.FORMAT_PEMBAYARAN_ID`. Label asli tetap disimpan di kolom
`src_format_pembayaran`.

| Asli | Jadi |
|---|---|
| Cheque | Cek |
| Wire | RTGS |
| ACH | Kliring |
| Cash | Tunai |
| Credit Card | Kartu Kredit |
| Bitcoin | **Transfer Valas** |

Lima baris pertama adalah pelokalan istilah. **Baris terakhir bukan** — ia
mengubah makna kanalnya dan wajib disebut setiap kali komposisi kanal dilaporkan.
Alasannya: `FACT_TRANSFER_GIRO` membingkai transfer sebagai giro **rupiah antar
entitas Indonesia** (kedua kaki dipetakan ke rekening debitur/counterparty
sintetis, mata uang asli tidak dibawa), sedangkan kripto tidak sah sebagai alat
pembayaran di Indonesia (UU No. 7/2011 tentang Mata Uang). `Transfer Valas`
dipilih karena sah, lazim di korporasi, dan mempertahankan peran asli `Bitcoin`
di dataset sebagai kanal lintas yurisdiksi berisiko tinggi - yang merupakan
sinyal AML-nya. Konsekuensinya: **jangan menarik kesimpulan tentang perilaku
kripto** dari kolom ini, dan share `Transfer Valas` bukan estimasi porsi
transaksi valas yang sebenarnya di populasi mana pun.

### 3.8 Waktu terjadinya default

`hari_ke_default` di SBA (selisih `ChgOffDate - DisbursementDate`) bermedian
**1.314 hari** (P10 694, P90 2.344) — jauh di luar jendela observasi 24 bulan
yang dipakai proyek ini.

Versi pertama pipeline memangkasnya (`min(hari, sisa_jendela)`). Akibatnya semua
default menumpuk di ujung jendela: umur default berkisar 321–693 hari dengan
simpangan baku 103, dan **bad rate 12 bulan jatuh ke 0,15%** (3 kejadian dari
2.036 fasilitas) sehingga PD 12 bulan tidak bisa dilatih sama sekali.

Yang dipakai sekarang adalah **penskalaan monoton berbasis peringkat**: persentil
`hari_ke_default` SBA dipetakan linier ke jendela 60–730 hari. Debitur yang di
data nyata gagal lebih cepat tetap gagal lebih cepat relatif terhadap yang lain,
tapi sebarannya mengisi seluruh jendela. Hasilnya median 274 hari (P10 107,
P90 481) dan bad rate 12 bulan 2,65%.

Fasilitas yang umur hasil pemetaannya melewati batas observasinya sendiri
**tidak dipaksa default** — ia menjadi observasi tersensor kanan. Itulah sebabnya
`FACT_DEFAULT` berisi 75 baris sementara 7,27% debitur berlabel gagal bayar:
sisanya belum sempat teramati.

Konsekuensi pelaporan: **urutan cepat/lambat gagal bayar bermakna, besaran
absolut harinya tidak.**

### 3.9 Porsi kepemilikan

ICIJ mencatat *siapa* pemegang saham, bukan *berapa persen*. Kolom
`FACT_KEPEMILIKAN.porsi_kepemilikan` karena itu SINTESIS (Dirichlet), sementara
keberadaan relasinya NYATA.

---

## 4. Cara penjahitan (langkah 1-7)

| Langkah | Isi | Modul |
|---|---|---|
| 1 | 6.000 perusahaan x 3 tahun berturut-turut dari `american_bankruptcy.csv` -> `CIF-000001...` | `joins.pilih_panel_debitur` |
| 2 | Satu baris rasio Taiwan per cif, dicocokkan **longgar**: label + kuintil DER + kuintil ROA | `joins.cocokkan_taiwan` |
| 3 | Satu simpul entitas ICIJ per cif, diambil per klaster hub pengurus | `joins.petakan_icij` |
| 4 | 1-3 rekening giro AML per cif, rekening berderajat tinggi ke debitur besar | `joins.petakan_rekening` |
| 5 | Satu baris SBA per cif (CHGOFF bila default, PIF bila tidak) -> `lgd_realisasi`, `Term`, `RevLineCr` | `joins.tarik_sba` |
| 6 | Generator sintesis mengisi sisanya | `generators/sintesis.py` |
| 7 | Injeksi afiliasi tersembunyi (45 klaster) | `generators/afiliasi.py` |
| 7b | Injeksi kekotoran data | **belum dikerjakan** |

### Langkah 7: injeksi afiliasi tersembunyi

Langkah 1 (label gagal bayar) dan langkah 3 (pemetaan simpul ICIJ) dikerjakan
independen satu sama lain, sehingga sebelum langkah 7 tidak ada jalur
sebab-akibat apa pun antara struktur graf dan label. Terukur: blok `graf_`
memberi AUC out-of-time 0,423 - di bawah tebakan acak.

Langkah 7 menanam jalur itu, dan dua hal harus benar sekaligus.

**Prasyarat - dua angkatan.** Fitur `neighbor_default_rate_1hop` hanya
menghitung gagal bayar yang tanggalnya SUDAH lewat pada snapshot. Dengan satu
angkatan (semua mengajukan 2025, semua jatuh sesudahnya), pada saat penilaian
belum ada satu pun kejadian: hanya 6 dari 3.000 pengajuan punya tetangga yang
pernah gagal bayar. Populasi karena itu dibelah menjadi buku lama (mengajukan
2022-2023, menghasilkan riwayat) dan buku baru (mengajukan 2025, bisa melihat
riwayat itu). Sesudahnya: 445 dari 6.000 pengajuan punya tetangga yang pernah
jatuh.

**Arah kausal.** Klaster dibentuk lebih dulu, lalu urutan waktu gagal bayarnya
diatur - bukan sebaliknya. Tiap klaster berisi 2 sumber (buku lama, jatuh
2022-2024), 2 terinfeksi (buku baru, jatuh sesudahnya), dan 4 anggota sehat.
Label tiap debitur tetap NYATA dari panel US; yang disintesis adalah struktur
ketergantungan dan urutan waktunya.

Mekanismenya menyamar sebagai relasi biasa - nominee bersama (`menjabat_di`),
alamat operasional bersama (`berbagi_atribut`), dan siklus pembayaran
(`memasok`). Edge hasil injeksi **tidak diberi penanda** di `GOLD_GRAPH_EDGES`:
kalau ditandai, ia tidak lagi tersembunyi. Ground truth-nya hidup terpisah di
`FACT_AFILIASI_TERSEMBUNYI` dan terdaftar di `gold.katalog_kolom_terlarang`.

### Kegagalan percobaan pertama: klaster menghabiskan kolam kejadian

Versi pertama membentuk 112 klaster dan menyerap **224 dari 229** debitur gagal
bayar buku baru. Rasio di dalam klaster tetap 2:4 seperti rancangan, tapi di
LUAR klaster tidak tersisa siapa-siapa:

| Peran | n | bad rate |
|---|---|---|
| bukan anggota | 1.629 | 0,003 |
| sehat | 314 | 0,000 |
| terinfeksi | 133 | 0,466 |

Keanggotaan klaster praktis menjadi label itu sendiri, dan model mengerjakan
soal yang salah.

Perbaikannya `afiliasi_porsi_default_terpakai = 0,40` - klaster hanya boleh
memakai 40% tiap kolam. Jumlah klaster turun ke 45, bad rate non-anggota kembali
ke 1,9%, dan dilusi keanggotaan terkunci di 33,3%.

Pelajarannya: menjaga rasio di dalam klaster tidak cukup. Yang menentukan adalah
berapa banyak kejadian yang TERSISA di luar.

### Catatan langkah 2 (korelasi palsu)

Pencocokan memakai **kuintil, bukan nilai**. Pada 6.000 debitur, mayoritas tercocokkan
pada kunci `label + kuintil DER + kuintil ROA` dan 20 jatuh ke fallback `label
saja`. Pencocokan yang lebih ketat akan menciptakan korelasi palsu yang meniup
AUC - metode ini harus dilaporkan bersama angka AUC mana pun.

**Kunci pencocokan memakai label, jadi rasio Taiwan membawa informasi target.**
Terukur: empat kolom `tw_*` saja menghasilkan AUC out-of-time 0,826, lebih
tinggi dari 29 kolom rasio nyata (0,662). Karena itu blok `tw_*` **dibuang dari
seluruh ABT** dan hanya tersisa di `FACT_LAPORAN_KEUANGAN` sebagai catatan
silsilah. Uji `test_blok_taiwan_tidak_kembali_ke_abt_pd` menjaganya.

### Catatan langkah 3 (kenapa bukan sampling acak)

Sampling acak 6.000 entitas dari 814.344 entitas ICIJ menghasilkan simpul yang
nyaris tidak saling terhubung, dan lapisan grafnya jadi kosong. Karena itu
entitas diambil per klaster hub: satu pengurus yang memegang 2-30 badan hukum
dianggap satu grup usaha, ditambah penggabungan lewat alamat domisili yang sama
(dibatasi maksimum 20 entitas per alamat supaya alamat agen registrasi tidak
menelan seluruh populasi).

Hasilnya: 6.000 debitur dalam **1,390 grup usaha**.
Konsekuensi: distribusi derajat tidak persis sama dengan populasi ICIJ penuh -
tersaring ke entitas yang punya minimal satu relasi pengurus.

### Catatan alamat (kenapa teksnya disintesis padahal strukturnya nyata)

`DIM_ALAMAT` memisahkan dua hal yang mudah tertukar:

- **Siapa sekantor dengan siapa** - diambil apa adanya dari ICIJ. Inilah sinyal
  yang dipakai untuk menemukan afiliasi lintas grup, dan ia nyata.
- **Teks alamatnya** - disintesis Faker. Alamat di berkas ICIJ adalah data nyata
  dari dokumen bocoran, dan proyek ini sudah menolak membawa nama asli ICIJ ke
  gold; alamat tunduk pada aturan yang sama. Alasan kedua: debitur di sini badan
  hukum Indonesia sintetis, dan alamat asli ICIJ membuat mereka berkantor di
  Sliema dan Tortola.

Penyamaran dikunci pada bentuk **ternormalisasi alamat aslinya**, jadi dua
entitas yang berbagi satu alamat di ICIJ tetap berbagi satu alamat sesudah
disamarkan. Uji `test_struktur_berbagi_alamat_bertahan_setelah_disamarkan`
menjaganya, dan `test_alamat_asli_icij_tidak_ikut_ke_gold` menjaga sisi
sebaliknya.

---

## 5. Aturan anti-bocor (§7.4)

1. Fitur graf untuk pengajuan bertanggal `T` dihitung pada snapshot **akhir bulan
   sebelum** `T`. Untuk pengajuan 2025-03-15, snapshot-nya 2025-02-28.
2. Edge masuk snapshot hanya bila `valid_from <= snapshot_date` dan
   (`valid_to` kosong atau `valid_to > snapshot_date`).
3. `community_default_rate` dan `neighbor_default_rate_1hop` hanya menghitung
   default yang `tanggal_default`-nya **sudah lewat** pada snapshot itu.
4. `src_is_laundering` hidup di `FACT_TRANSFER_GIRO` dan **tidak pernah** masuk
   `FEAT_GRAF_PIT`. Kolom itu hanya untuk mengevaluasi `circular_payment_flag`
   dan Isolation Forest (§7.2-D), bukan fitur model PD.
5. `FEAT_GRAF_PIT` terpisah fisik dari `FACT_LAPORAN_KEUANGAN`, sehingga uji
   ablasi §7.3 (model tanpa vs dengan blok graf) bisa dilakukan dengan
   men-drop satu tabel utuh.

Kelima aturan ini diuji di `pipelines/quality/checks.py` dan
`pipelines/tests/test_gold_layer.py`. Aturan 1 pernah dilanggar oleh bug helper
tanggal dan **ditangkap oleh uji tersebut**, bukan oleh review manual.

---

## 6. Angka hasil eksekusi (seed 42, N_DEBITUR=6000)

Populasi dinaikkan dari 3.000 ke 6.000 debitur (dari 6.251 perusahaan layak)
karena setelah dibagi dua angkatan, 3.000 hanya menyisakan 3 kejadian gagal
bayar di uji out-of-time. Ini melampaui rentang 8.000-12.000 firm-year pada
rencana data - pilihan sadar, tercatat alasannya di `config.py`.

### ERD A - inti kredit

| Tabel gold | Baris |
|---|---|
| `dim_debitur` | 6,842 (6.000 versi kini + sisanya SCD-2) |
| `dim_grup_usaha` | 1,390 |
| `fact_laporan_keuangan` | 18,000 |
| `fact_pengajuan` | 6,000 |
| `fact_fasilitas` | 4,398 |
| `fact_agunan` | 8,861 |
| `fact_covenant` | 388,740 |
| `fact_kolektibilitas` | 129,580 |
| `fact_default` | 215 (sisanya tersensor kanan) |
| `fact_eksposur_grup` | 55,522 |

### ERD B - lapisan graf

| Tabel gold | Baris |
|---|---|
| `gold_graph_nodes` | 37,170 |
| `gold_graph_edges` | 88,500 |
| `fact_transfer_giro` | 297,553 |
| `graph_snapshot_bulanan` | 1,338,120 (36 snapshot) |
| `feat_graf_pit` | 6,000 |
| `fact_afiliasi_tersembunyi` | 360 (45 klaster) |

### Paket serah terima (ABT)

| Tabel | Baris |
|---|---|
| `abt_pd` | 4,398 (4.254 dapat dilatih, 144 tersensor) |
| `abt_ews` | 126,127 (104.256 dapat dilatih) |
| `abt_lgd` | 215 |
| `abt_lgd_sumber` | 156,824 |
| `abt_pengajuan_ditolak` | 1,602 |
| `kamus_data_abt` | 241 |

Tingkat gagal bayar debitur 7,6%; bad rate PD 12 bulan 3,2% (136 kejadian);
bad rate EWS 6 bulan 1,15%; rasio transfer ilisit 0,0474% (sumber LI-Small
~0,05%); LGD rata-rata pada SBA CHGOFF 0,619.

Cara memakai ABT ada di
[serah-terima-data-scientist.md](serah-terima-data-scientist.md).
