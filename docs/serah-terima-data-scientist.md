# Serah Terima Data untuk Data Scientist

Dokumen ini yang dibaca duluan sebelum menyentuh datanya.

---

## 0. Baca ini dulu

Data ini menggabungkan tujuh dataset publik. **Join antar dataset itu tidak ada
di dunia nyata** — baris rasio Taiwan tidak berhubungan dengan akun AML maupun
simpul ICIJ; semuanya ditempelkan pada satu `cif` sintetis. Sinyal dan
topologinya nyata, keterkaitannya tidak.

Konsekuensi praktis: **kalau AUC keluar di atas 0,90, itu hampir pasti artefak,
bukan prestasi.** Lapor angkanya bersama metode pencocokannya. Rinciannya di
[data-lineage.md](data-lineage.md).

---

## 1. Apa yang diserahkan

Semuanya di `data/gold/` (parquet), plus kamusnya:

| Berkas | Grain | Baris | Target |
|---|---|---|---|
| `abt_pd.parquet` | `application_id` | 4.398 | `y_default_12bln` |
| `abt_ews.parquet` | `facility_id` × `snapshot_date` | 126.127 | `y_default_6bln` |
| `abt_lgd.parquet` | `facility_id` yang default | 215 | `y_lgd_realisasi` |
| `abt_lgd_sumber.parquet` | pinjaman SBA CHGOFF | 156.824 | `y_lgd_realisasi` |
| `abt_pengajuan_ditolak.parquet` | `application_id` | 1.602 | — (tanpa outcome) |
| `kamus_data_abt.csv` | — | 241 | kamus kolom seluruh ABT |

Cara baca:

```python
import pandas as pd
abt = pd.read_parquet("data/gold/abt_pd.parquet")
kamus = pd.read_csv("data/gold/kamus_data_abt.csv")
```

Kalau lebih suka SQL, jalankan `LOAD_TO_POSTGRES=1` lalu tabel yang sama ada di
skema `gold` Postgres.

---

## 2. Struktur kolom

Setiap kolom fitur punya prefiks blok. Ini bukan kosmetik — inilah yang membuat
uji ablasi §7.3 jadi satu baris.

| Prefiks | Isi | Jumlah di `abt_pd` |
|---|---|---|
| `fin_` | rasio keuangan tahun buku 2024 + tren 3 tahun (**NYATA**) | 29 |
| `app_` | plafon, tenor, produk, agunan, rating, profil debitur pada saat T | 23 |
| `graf_` | fitur graf titik-waktu | 28 |
| `y_` | target dan penanda sensor | 4 |
| tanpa prefiks | kunci, tanggal, `angkatan`, `split` | 9 |

```python
# Model tanpa blok graf (baseline §7.3)
X_baseline = X.drop(columns=X.filter(like="graf_").columns)
```

---

## 3. Target dan sensor kanan

`y_default_12bln` punya **tiga** nilai, bukan dua:

| Nilai | Artinya |
|---|---|
| `1.0` | default terjadi dalam 12 bulan sejak pencairan |
| `0.0` | bertahan 12 bulan penuh, dan 12 bulan itu benar-benar teramati |
| `NaN` | **tersensor kanan** — jendela 12 bulannya belum genap saat data berakhir |

Baris `NaN` **jangan diisi 0.** 144 dari 4.398 baris tersensor; menandainya 0
akan menurunkan bad rate secara palsu. Buang untuk model klasifikasi, atau pakai
`y_umur_teramati_hari` + `y_tersensor` untuk model survival (Cox / AFT).

```python
latih = abt[(abt.split == "latih") & abt.y_default_12bln.notna()]
uji   = abt[(abt.split == "uji_oot") & abt.y_default_12bln.notna()]
```

Kolom `angkatan` membelah populasi: `buku_lama` (mengajukan 2022-2023) hanya ada
untuk menghasilkan riwayat gagal bayar, dan fitur penularan grafnya kosong karena
belum ada apa pun sebelum mereka. **Untuk mengukur kontribusi blok graf, saring
ke `angkatan == "buku_baru"`.**

Angka populasinya:

**Seluruh populasi** (buku lama + buku baru):

| Split | Baris | Dapat dilatih | Kejadian default | Bad rate |
|---|---|---|---|---|
| `latih` (s.d. 30 Sep 2025) | 3.859 | 3.859 | 122 | 3,16% |
| `uji_oot` (Q4 2025) | 539 | 395 | 14 | 3,54% |

**Hanya `buku_baru`** — inilah populasi untuk mengukur kontribusi blok graf:

| Split | Baris | Dapat dilatih | Kejadian default |
|---|---|---|---|
| `latih` | 1.681 | 1.681 | 53 |
| `uji_oot` | 539 | 395 | 14 |

Kelasnya tidak seimbang (~3,2%). Pakai `scale_pos_weight` / `class_weight`,
jangan oversampling naif di seluruh data (itu membocorkan antar fold).

**14 kejadian di uji out-of-time itu sedikit.** Selisih AUC di bawah ~0,05 pada
populasi sebesar ini tidak bisa dibedakan dari derau — sertakan selang
kepercayaan bootstrap setiap kali melaporkan perbandingan model.

---

## 3b. Cara memperlakukan NaN

`abt_pd` punya 11 kolom ber-NaN dengan **empat sebab berbeda**. Perlakuannya
tidak sama, dan salah satunya menghapus sinyal kalau ditangani asal. Kolom
`catatan` di `kamus_data_abt.csv` menandai semuanya.

| Kolom | NaN | Artinya | Perlakuan |
|---|---|---|---|
| `y_umur_hari` | 95,1% | hanya terisi bila gagal bayar | pakai `y_umur_teramati_hari` untuk survival |
| `fin_debt_to_ebitda*`, `fin_cfo_to_ebitda` | 25–33% | **EBITDA ≤ 0** | indikator kosong + imputasi, atau biarkan |
| `graf_supplier/buyer_concentration_hhi` | 29% / 14% | tidak ada relasi transfer | `fillna(0)` |
| `graf_neighbor_default_rate_1hop` | 9,2% | tidak punya tetangga debitur | `fillna(0)` |
| `y_default_12bln` | 3,3% | tersensor kanan | buang, jangan diisi 0 |
| `fin_dio_hari`, `fin_siklus_modal_kerja_hari` | 0,27% | COGS = 0 | imputasi biasa |

### Yang paling mudah salah: `fin_debt_to_ebitda`

1.121 debitur punya EBITDA nol atau negatif, sehingga rasionya memang tidak
terdefinisi. **NaN-nya bukan data hilang — ia keadaan, dan keadaan itu
informatif:**

| | n | bad rate |
|---|---|---|
| `fin_debt_to_ebitda` NaN | 1.085 | **6,3%** |
| terisi | 3.169 | 2,1% |

Tiga kali lipat. `fillna(median)` menghapus sinyal itu **dan** memberi debitur
ber-EBITDA negatif angka rasio yang tampak sehat — dua kerugian sekaligus.

```python
# Pertahankan keadaannya sebelum mengimputasi
X["fin_ebitda_nonpositif"] = X.fin_debt_to_ebitda.isna().astype(int)
X["fin_debt_to_ebitda"] = X.fin_debt_to_ebitda.fillna(X.fin_debt_to_ebitda.median())
```

Atau pakai model yang menangani NaN sendiri (LightGBM, XGBoost) dan biarkan apa
adanya. Keduanya jauh lebih baik daripada imputasi polos.

### Kolom `graf_`: NaN berarti nol, bukan tidak diketahui

Konsentrasi pemasok kosong karena pada snapshot itu debitur memang tidak punya
edge transfer sama sekali. Di sini `fillna(0)` **benar secara semantik**;
`fillna(median)` mengarang relasi yang tidak ada. Bad rate-nya nyaris sama
(3,4% vs 3,1%), jadi tidak informatif — beda dengan kasus EBITDA di atas.

---

## 4. Split: pakai yang sudah disediakan

Kolom `split` sudah out-of-time: pengajuan sampai 30 September 2025 untuk latih,
sesudahnya untuk uji.

**Jangan pakai `train_test_split` acak.** Satu grup usaha (`grup_id`) punya
beberapa debitur yang berbagi tetangga graf dan alamat; split acak menaruh
anggota grup yang sama di kedua sisi dan membocorkan `graf_*`. Kalau butuh
cross-validation di dalam data latih, pakai `GroupKFold(groups=abt.grup_id)`.

---

## 5. Aturan yang sudah dijaga pipeline

Yang berikut ini sudah diuji otomatis, tidak perlu Anda cek ulang:

- `graf_snapshot_date` selalu akhir bulan **sebelum** bulan pengajuan.
- Tidak ada edge dengan `valid_from > snapshot_date` yang ikut terhitung.
- `graf_community_default_rate` dan `graf_neighbor_default_rate_1hop` hanya
  menghitung default yang tanggalnya sudah lewat pada snapshot itu.
- `abt_pd` tidak memuat kolom pasca-pencairan (`outstanding`, `pemakaian_plafon`,
  `kolektibilitas`, `dpd`, `tanggal_default`).
- `abt_ews` tidak memuat baris setelah fasilitasnya default.
- Tidak ada `label_default`, `label_default_debitur`, `status_label`, atau
  `src_is_laundering` di ABT mana pun.

Uji-ujinya ada di `pipelines/tests/test_abt.py` dan `pipelines/quality/checks.py`
(43 uji pytest + 69 uji gerbang kualitas, semuanya lolos). Jalankan `python -m pytest` kalau Anda mengubah ABT.

---

## 6. Yang HARUS Anda putuskan sendiri

**`app_keputusan` dan `app_pricing_bps` adalah hasil proses keputusan, bukan
input.** Keduanya ikut diekspor karena berguna untuk analisis kebijakan, tapi
kalau model Anda memprediksi PD untuk *mendukung* keputusan, keduanya harus
di-drop — kalau tidak, model belajar meniru aturan persetujuan sintetis, bukan
belajar risiko.

```python
X = X.drop(columns=["app_keputusan", "app_pricing_bps", "app_komite_level"])
```

**`app_skor_kredit` dan `app_rating_internal`** diturunkan dari peringkat rasio
nyata (DER, ICR, ROA, debt/EBITDA) — tidak menyentuh label. Aman sebagai fitur,
tapi berkorelasi tinggi dengan blok `fin_`; jangan kaget kalau ia mendominasi
importance.

**Populasi `abt_pd` hanya pengajuan yang cair.** Yang ditolak (1.602 baris) ada di
`abt_pengajuan_ditolak.parquet` tanpa target. Kalau modelnya untuk *screening*
pengajuan baru, ada bias seleksi yang perlu ditangani (reject inference) — data
untuk itu sudah disediakan, metodenya keputusan Anda.

---

## 7. Kesiapan tiap ABT (hasil audit sinyal)

Tiap blok fitur diuji dengan regresi logistik sederhana, dilatih di `latih` dan
diukur di `uji_oot`. Angka ini bukan target performa — gunanya menunjukkan blok
mana yang benar-benar membawa sinyal.

### `abt_pd` — siap, dengan satu catatan besar

| Blok | Kolom | AUC latih | AUC uji OOT |
|---|---|---|---|
| `fin_` (rasio panel US) | 29 | 0,804 | 0,662 |
| `app_` (pengajuan + profil) | 16 | 0,870 | 0,742 |
| `graf_` (fitur graf) | 28 | 0,702 | **0,423** |
| `fin_` + `app_` (baseline) | 45 | 0,912 | 0,708 |
| baseline + `graf_` | 73 | 0,928 | 0,667 |

**Angka di atas adalah kondisi SEBELUM langkah 7.** Blok graf menurunkan AUC
karena pemetaan simpul ICIJ dikerjakan independen dari label - secara konstruksi
tidak ada jalur sebab-akibat untuk dipelajari.

Setelah langkah 7 (injeksi afiliasi tersembunyi) dan pembagian dua angkatan,
ukur ulang pada populasi `angkatan == "buku_baru"`:

| Model | AUC latih | AUC uji OOT |
|---|---|---|
| baseline (`fin_` + `app_`) | 0,881 | 0,733 |
| baseline + blok graf | 0,899 | **0,754** |
| penularan masa lalu saja (2 kolom) | 0,581 | 0,490 |
| topologi tanpa penularan | 0,669 | 0,617 |

**Selisihnya +0,021, dan selang kepercayaan 95% bootstrap-nya [-0,027, +0,071] -
mencakup nol.** Dengan 14 kejadian gagal bayar di uji out-of-time, lift ini
**tidak terukur secara statistik**. Jangan laporkan "fitur graf meningkatkan
AUC" berdasarkan angka ini.

Yang terukur adalah perbedaan risikonya, bukan AUC-nya:

| | n | bad rate |
|---|---|---|
| `graf_neighbor_default_rate_1hop > 0` | 204 | **6,4%** |
| `= 0` | 1.872 | 2,9% |

Selisih 2,2x, dan arahnya benar. Untuk mengubahnya menjadi lift AUC yang bisa
dipertanggungjawabkan, yang dibutuhkan adalah lebih banyak kejadian di jendela
uji - bukan parameter injeksi yang dinaikkan sampai grafiknya bagus.

Blok `fin_tw_*` (rasio Taiwan) **sudah dibuang** dari ABT. Empat kolom itu
menghasilkan AUC out-of-time 0,826 — lebih tinggi dari 29 kolom rasio nyata —
karena baris Taiwan-nya dicocokkan **memakai label gagal bayar** (langkah 2:
kunci = label + kuintil DER + kuintil ROA). Itu kebocoran lewat kunci
pencocokan, bukan sinyal keuangan.

### `abt_ews` — siap untuk populasi kol 1–2 saja

`perilaku_kolektibilitas` awalnya adalah fungsi deterministik dari jarak ke
tanggal default: `P(default 6 bulan | kol = 3)` tepat **1,000**. Model EWS akan
menghafal aturan generator, bukan belajar.

Generator sudah diperbaiki — kini ada episode tekanan yang pulih (*cured*) pada
fasilitas yang tidak berakhir default, dan derau pada ramp menjelang default:

| Kolektibilitas | P(default dalam 6 bulan) |
|---|---|
| 1 | 0,007 |
| 2 | 0,066 |
| 3 | **0,277** (sebelumnya 1,000) |
| 4 | 1,000 |

Kol 4 masih deterministik, dan itu wajar — kol 4 berarti sudah 121–180 DPD,
praktis sudah bermasalah. **Populasi EWS yang bermakna adalah kol 1–2**, persis
seperti di bank sungguhan: nilai sebuah EWS ada pada kemampuannya menandai
rekening yang *masih terlihat sehat*. Latih di sana, dan laporkan performa
terpisah untuk kol 1–2.

### `abt_pengajuan_ditolak` — bukan data latih

1.602 baris tanpa target, jadi tidak bisa dilatih. Gunanya untuk reject inference.
Ruang fiturnya kini **identik dengan `abt_pd`** (86 kolom, tanpa blok `y_`),
sehingga model yang sama bisa menskor kedua populasi. Kolom yang lahir dari
fasilitas (agunan, plafon final, tanggal pencairan) memang tidak ada dan
bernilai NA — itu keadaan sebenarnya, bukan data hilang.

### `abt_lgd` — untuk menerapkan, bukan melatih

Lihat §1 dan kolom `catatan` di kamus data. Latih di `abt_lgd_sumber`
(156.824 pinjaman SBA nyata, split OOT berbasis `ApprovalFY`), lalu terapkan
hasilnya ke `abt_lgd` untuk menghitung `PD x LGD x EAD`.

Di `abt_lgd`, hanya kolom yang berasal dari baris SBA yang sama dengan target
yang punya hubungan nyata dengan LGD (`app_produk_id`, `app_jenis_fasilitas`,
`app_tenor_bulan_sba`, `app_porsi_penjaminan`, `app_skala_pegawai`,
`app_dokumen_ringkas`, `app_perusahaan_baru`, `app_sektor_kbli_sba`). Sisanya —
termasuk `app_coverage_ratio` — derau. Kamus data menandai keduanya secara
eksplisit di kolom `catatan`.

---

## 8. Batasan yang perlu dilaporkan bersama hasil

1. **LGD portofolio hanya 215 observasi**, dan hanya sebagian fiturnya bersinyal.
   Model LGD dilatih di `abt_lgd_sumber` (156.824 pinjaman SBA nyata), bukan di
   sini. Nilai LGD-nya nyata (dari `ChgOffPrinGr / DisbursementGross`), tapi
   fasilitas yang dilekatinya sintetis.
2. **Waktu default hasil penskalaan.** `hari_ke_default` asli SBA bermedian
   1.314 hari; diskala monoton ke jendela 60–730 hari. Urutan cepat/lambat
   dipertahankan, besaran absolutnya tidak berarti.
3. **ICR memakai asumsi tarif pajak 25%** karena beban bunga tidak ada di data
   sumber. Lihat [data-lineage.md §3.1](data-lineage.md).
4. **Siklus modal kerja = DSO + DIO tanpa DPO** (utang usaha tidak tersedia),
   jadi angkanya lebih tinggi dari siklus kas sebenarnya.
5. **`graf_` tipis untuk sebagian debitur** (~22% pengajuan tanpa relasi pemasok
   pada snapshot-nya) **dan netral terhadap target** — lihat §7. Perlakukan
   `NaN`-nya sebagai "tidak ada relasi", bukan "tidak diketahui".
6. **`src_is_laundering` sengaja tidak diberikan di ABT.** Kolom itu ada di
   `fact_transfer_giro` dan hanya untuk mengevaluasi deteksi anomali struktural
   (§7.2-D), bukan fitur PD.
