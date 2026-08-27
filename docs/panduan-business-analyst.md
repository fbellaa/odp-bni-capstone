# Panduan Data untuk Tim Business Analyst

Dokumen ini untuk yang membangun dashboard, laporan portofolio, dan analisis
eksposur di atas `data/gold/`. Kalau Anda melatih model, yang Anda butuhkan
adalah [serah-terima-data-scientist.md](serah-terima-data-scientist.md).

---

## 0. Aturan pertama: angka mana yang boleh dikutip

Data ini menggabungkan tujuh dataset publik, dan **keterkaitan antar sumbernya
sintetis**. Untuk Anda yang membuat slide dan laporan, konsekuensinya sangat
praktis:

| Boleh dikutip sebagai temuan | JANGAN dikutip sebagai temuan |
|---|---|
| Bentuk sebaran rasio keuangan | Nominal rupiah apa pun |
| Tingkat gagal bayar & tingkat pemulihan | "Total portofolio Rp 93,6 T" |
| Struktur kepemilikan berlapis, rangkap jabatan | Nama PT, alamat, NPWP |
| Pola topologi transfer (fan-in/out, siklus) | Sektor mana paling berisiko |
| Perilaku migrasi kolektibilitas | Nilai agunan, pricing, BMPK |

**Semua rupiah di sini hasil penskalaan sintetis.** Peringkat besar-kecil antar
debitur mengikuti data nyata, tapi angkanya sendiri tidak berarti apa-apa. Kalau
sebuah angka rupiah muncul di slide, wajib disertai label "data sintetis".

Sektor KBLI juga turunan sintetis (pemetaan dari NAICS Amerika), jadi
kesimpulan seperti "sektor perdagangan paling berisiko" **tidak sah** — sektornya
tidak berhubungan dengan risiko debitur di data ini.

Rinciannya di [data-lineage.md](data-lineage.md).

---

## 1. Tabel mana untuk pertanyaan apa

| Pertanyaan bisnis | Tabel utama |
|---|---|
| Profil debitur, sektor, rating, skala usaha | `dim_debitur` |
| Rasio keuangan & trennya | `fact_laporan_keuangan` |
| Berapa pengajuan masuk, berapa disetujui | `fact_pengajuan` |
| Baki debet, plafon, pemakaian | `fact_fasilitas`, `fact_kolektibilitas` |
| Kualitas aset / NPL per bulan | `fact_kolektibilitas` |
| Pelanggaran covenant | `fact_covenant` |
| Eksposur grup vs BMPK | `fact_eksposur_grup`, `dim_grup_usaha` |
| Kerugian & pemulihan | `fact_default` |
| Struktur grup, pemilik, pengurus | `dim_pihak`, `fact_kepemilikan`, `fact_kepengurusan` |
| Jaringan pemasok/pembeli | `gold_graph_edges`, `dim_counterparty` |
| Hasil uji kualitas data | `data_quality_report` |

Semuanya tersedia sebagai parquet di `data/gold/`, atau sebagai tabel SQL di
skema `gold` Postgres setelah `LOAD_TO_POSTGRES=1`.

---

## 2. Lima jebakan agregasi

Ini bagian terpenting dokumen ini. Keempat jebakan pertama menghasilkan angka
yang **kelihatan wajar tapi salah besar**.

### Jebakan 1 — `dim_debitur` punya versi ganda

Tabel ini SCD-2: satu debitur bisa punya dua baris kalau rating internalnya
pernah berubah.

```sql
SELECT COUNT(*) FROM gold.dim_debitur;                    -- 6.842  SALAH
SELECT COUNT(*) FROM gold.dim_debitur WHERE is_current;   -- 6.000  benar
```

Tanpa `WHERE is_current`, jumlah debitur kelebihan **842**. Selalu saring, kecuali
Anda memang sedang menganalisis migrasi rating.

### Jebakan 2 — kolektibilitas adalah snapshot bulanan

`fact_kolektibilitas` berisi satu baris per fasilitas **per bulan** (60 bulan).
Menjumlahkan seluruh baris berarti menghitung baki debet yang sama puluhan kali.

```sql
-- SALAH: Rp 3.213 T (34x lipat)
SELECT SUM(outstanding_rp) FROM gold.fact_kolektibilitas;

-- BENAR: Rp 93,6 T
SELECT SUM(outstanding_rp) FROM gold.fact_kolektibilitas
WHERE snapshot_date = '2026-12-31';
```

Aturannya: **untuk nilai posisi (baki debet, eksposur), selalu pilih satu
`snapshot_date`.** Untuk tren, kelompokkan per `snapshot_date` — jangan pernah
menjumlahkan lintas bulan.

Berlaku sama untuk `fact_covenant`, `fact_eksposur_grup`, dan
`graph_snapshot_bulanan`.

### Jebakan 3 — ada dua angkatan, dan 2024 kosong

Populasi sengaja dibelah dua:

| Angkatan | Tahun pengajuan | Jumlah |
|---|---|---|
| `buku_lama` | 2022 (1.524), 2023 (1.476) | 3.000 |
| `buku_baru` | 2025 | 3.000 |

**Tidak ada pengajuan di 2024.** Grafik "pengajuan per tahun" akan menunjukkan
lubang, dan itu bukan anomali bisnis — itu rancangan, supaya buku lama sempat
menghasilkan riwayat gagal bayar sebelum buku baru mengajukan.

Kalau dashboard Anda menampilkan tren tahunan, beri catatan, atau pisahkan
kedua angkatan.

### Jebakan 4 — satu pengajuan belum tentu jadi fasilitas

`fact_pengajuan` 6.000 baris, `fact_fasilitas` 4.398. Selisihnya pengajuan yang
ditolak. Untuk tingkat persetujuan gunakan `fact_pengajuan.keputusan`; untuk
apa pun yang menyangkut uang gunakan `fact_fasilitas`.

### Jebakan 5 — `grup_id` bukan satu-satunya keterkaitan

`dim_grup_usaha` adalah grup yang **kasat mata**. Di data ini juga ditanam
**afiliasi tersembunyi** yang sengaja melintasi grup (`fact_afiliasi_tersembunyi`,
45 klaster). Laporan eksposur berbasis `grup_id` saja karena itu
**meremehkan** konsentrasi sebenarnya — dan itu memang inti masalah bisnis yang
hendak ditunjukkan produk ini.

`fact_afiliasi_tersembunyi` adalah kunci jawaban untuk mengevaluasi deteksi.
Jangan dipakai sebagai input laporan yang seolah-olah bank sudah mengetahuinya.

Yang **boleh** dipakai sebagai input laporan adalah `dim_alamat` dan
`fact_alamat_debitur`. Dua tabel itu berisi alamat operasional yang memang
diketahui bank dari dokumen domisili usaha, dan dari sanalah keterkaitan lintas
grup bisa ditemukan secara sah:

```sql
-- Debitur yang berbagi alamat operasional TAPI beda grup usaha.
SELECT a.alamat_id, a.alamat_teks,
       COUNT(DISTINCT d.grup_id) AS jumlah_grup,
       COUNT(DISTINCT d.cif_sk)  AS jumlah_debitur
FROM gold.dim_alamat a
JOIN gold.fact_alamat_debitur fa ON fa.alamat_id = a.alamat_id
JOIN gold.dim_debitur d          ON d.cif_sk = fa.cif_sk AND d.is_current
WHERE NOT a.is_alamat_agen
GROUP BY a.alamat_id, a.alamat_teks
HAVING COUNT(DISTINCT d.grup_id) > 1
ORDER BY jumlah_debitur DESC;
```

`is_alamat_agen` menandai alamat yang dipakai lebih dari 20 debitur — itu kantor
agen registrasi, bukan tanda keterkaitan usaha. **Selalu saring kolom itu**,
kalau tidak laporan Anda akan mengaku menemukan satu "grup" berisi ratusan
badan hukum yang cuma sekantor notaris.

Alamat pada `dim_alamat` adalah alamat **sintetis**. Alamat asli ICIJ tidak
dibawa ke gold karena itu data nyata dari dokumen bocoran; yang dipertahankan
hanya strukturnya — dua debitur yang berbagi alamat di sumber tetap berbagi
alamat di sini.

---

## 3. Resep siap pakai

**Posisi portofolio pada satu tanggal**

```sql
SELECT
    k.snapshot_date,
    COUNT(DISTINCT k.facility_id)          AS jumlah_fasilitas,
    SUM(k.outstanding_rp)                  AS baki_debet_rp,
    SUM(CASE WHEN k.kolektibilitas >= 3
             THEN k.outstanding_rp ELSE 0 END)
        / NULLIF(SUM(k.outstanding_rp), 0) AS rasio_npl
FROM gold.fact_kolektibilitas k
WHERE k.snapshot_date = DATE '2026-12-31'
GROUP BY k.snapshot_date;
```

**Tren NPL bulanan**

```sql
SELECT snapshot_date,
       SUM(CASE WHEN kolektibilitas >= 3 THEN outstanding_rp ELSE 0 END)
           / NULLIF(SUM(outstanding_rp), 0) AS rasio_npl
FROM gold.fact_kolektibilitas
GROUP BY snapshot_date
ORDER BY snapshot_date;
```

**Sepuluh grup dengan pemakaian BMPK tertinggi**

```sql
SELECT g.nama_grup, g.jumlah_entitas,
       e.total_eksposur_rp, e.group_exposure_share, e.sisa_ruang_rp
FROM gold.fact_eksposur_grup e
JOIN gold.dim_grup_usaha g USING (grup_id)
WHERE e.snapshot_date = DATE '2026-12-31'
ORDER BY e.group_exposure_share DESC
LIMIT 10;
```

**Tingkat persetujuan per kelas penjualan**

```sql
SELECT d.kelas_penjualan,
       COUNT(*) AS pengajuan,
       AVG(CASE WHEN p.keputusan <> 'tolak' THEN 1.0 ELSE 0 END) AS tingkat_setuju
FROM gold.fact_pengajuan p
JOIN gold.dim_debitur d ON d.cif_sk = p.cif_sk AND d.is_current
GROUP BY d.kelas_penjualan
ORDER BY pengajuan DESC;
```

**Pelanggaran covenant terkini**

```sql
SELECT c.jenis, COUNT(*) AS jumlah,
       AVG(CASE WHEN c.status = 'langgar' THEN 1.0 ELSE 0 END) AS rasio_langgar
FROM gold.fact_covenant c
WHERE c.snapshot_date = DATE '2026-12-31'
GROUP BY c.jenis;
```

Versi pandas untuk semuanya sama polanya:

```python
import pandas as pd
k = pd.read_parquet("data/gold/fact_kolektibilitas.parquet")
akhir = k[k.snapshot_date == k.snapshot_date.max()]
npl = akhir.loc[akhir.kolektibilitas >= 3, "outstanding_rp"].sum() / akhir.outstanding_rp.sum()
```

---

## 4. Angka acuan (posisi 31 Desember 2026)

Pakai ini untuk memeriksa apakah query Anda benar. Kalau hasil Anda jauh
melenceng, kemungkinan besar terkena salah satu jebakan di §2.

| Metrik | Nilai |
|---|---|
| Debitur aktif | 6.000 |
| Grup usaha | 1.390 |
| Fasilitas aktif pada snapshot terakhir | 3.776 |
| Baki debet | Rp 93,6 T *(sintetis)* |
| Rasio NPL (kol 3–5) | 3,78% |
| Sebaran kolektibilitas | kol 1: 3.411 · kol 2: 223 · kol 3: 1 · kol 5: 141 |
| Pengajuan | 6.000 (4.398 cair, 1.602 ditolak) |
| Fasilitas gagal bayar | 215 |
| Grup melewati BMPK | 0 |

---

## 5. Anomali yang sudah diketahui — jangan dijadikan headline

Tiga hal di bawah adalah **artefak generator sintetis**, bukan temuan bisnis.
Kalau muncul di dashboard, beri catatan atau jangan ditonjolkan.

**Pelanggaran covenant 40%.** Pada snapshot terakhir, 4.531 dari 11.328 posisi
covenant berstatus `langgar`. Portofolio bank sungguhan berada di kisaran 5–15%.
Ambangnya diturunkan dari kelas rating dengan aturan sederhana, sementara rasio
aktualnya berjalan acak — kombinasi itu menghasilkan pelanggaran jauh lebih
sering daripada semestinya.

**Tidak ada satu pun grup melewati BMPK.** Batasnya ditetapkan 25% dari modal
bank sintetis Rp 12 T, dan tidak ada grup yang mendekatinya. Dashboard "grup
melampaui BMPK" akan selalu kosong. Kalau butuh kasus pelanggaran untuk
demonstrasi, batasnya harus diturunkan di `generators/sintesis.py` — dan
perubahan itu harus disebut di slide.

**Kolektibilitas 4 kosong pada snapshot terakhir.** Fasilitas yang gagal bayar
bertahan di kolektibilitas 5 sampai akhir jendela observasi, jadi kol 4 hanya
muncul sesaat sebelum gagal bayar. Sebaran kolektibilitas di sini tidak
mencerminkan sebaran portofolio bank sungguhan.

---

## 6. Kalau angkanya berubah

Data ini deterministik: seed yang sama menghasilkan berkas yang sama persis.
Kalau angka di dashboard Anda tiba-tiba bergeser padahal query-nya tidak
berubah, artinya seseorang menjalankan ulang pipeline dengan parameter berbeda —
paling sering `N_DEBITUR` atau porsi angkatan di `pipelines/config.py`.

Periksa dua tabel:

- **`gold.parameter_build`** — parameter efektif build ini (`n_debitur`, `seed`,
  `git_commit`, waktu bangun). Kolom `sumber` menandai parameter yang ditimpa
  berkas `.env`; itulah yang biasanya menjelaskan kenapa angka berubah.
- **`gold.data_quality_report`** — hasil uji kualitas beserta `dijalankan_pada`.

Pipeline-nya sendiri deterministik: seed sama menghasilkan berkas identik byte
per byte. Kalau angka bergeser, yang berubah parameternya, bukan pipeline-nya.

Pertanyaan soal definisi kolom: `gold.kamus_data_abt` memuat kamus per kolom,
dan `gold.katalog_kolom_terlarang` mendaftar kolom yang **tidak boleh** dipakai
sebagai bahan analisis karena merupakan kunci jawaban evaluasi.
