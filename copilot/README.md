# Lapisan copilot

LLM lokal untuk membaca dokumen pengajuan, RAG atas korpus kebijakan, dan agen
tool calling untuk perhitungan kreditnya.

```
copilot/
├─ konfigurasi.py   profil model, anggaran memori, path index
├─ llm/             klien Ollama (chat, streaming, JSON terstruktur, embedding)
├─ dokumen/         PDF -> struktur -> argumen telusuri_afiliasi()
├─ rag/             potong per pasal -> embed -> cari, dengan sitasi
├─ alat/            12 tool perhitungan deterministik + definisi function calling
├─ agen/            putaran tool calling
├─ memo.py          penyusunan draft credit memo
└─ scripts/         penyiapan Ollama dan uji asap
```

Antarmukanya ada di halaman **7 · Copilot lokal** pada aplikasi Streamlit.

---

## Rantai kerjanya

```
PDF pengajuan
   │  pypdf                          copilot/dokumen/pdf.py
   ▼
teks per halaman + tebakan jenis dokumen
   │  LLM peran `ekstraksi`, keluaran divalidasi Pydantic
   ▼
BerkasPengajuan (rekening koran, lapkeu, akta)
   ├─────────────► telusuri_afiliasi()          pipelines/graph/resolusi.py
   │                 alamat, pengurus, rekening lawan
   ├─────────────► RAG kebijakan                copilot/rag/
   │                 potongan per pasal + sitasi halaman
   ▼
konteks_pengajuan()  — fakta datar, satu angka per baris
   │  LLM peran `agen` (Qwen), function calling
   ▼
12 tool perhitungan  — SELURUH angka lahir di sini
   ▼
draft credit memo    copilot/memo.py
```

## Pembagian peran yang dijaga

| Lapisan | Boleh | Tidak boleh |
|---|---|---|
| Model bahasa (peran `ekstraksi`, `chat`) | menyalin angka dari PDF, menulis narasi | menghitung apa pun |
| Model agen (peran `agen`) | memilih tool, mengisi argumen, merangkai hasil | menuliskan angka hasil hitungannya sendiri |
| `copilot/alat/` | seluruh aritmetika | memanggil model |

Karena itu tiap tool mengembalikan field `rumus` yang menuliskan perhitungannya
apa adanya, dan tiap angka di memo mencantumkan nama tool yang menghasilkannya.
Angka yang tidak punya jejak tool tidak boleh masuk memo.

**Belum ada tool model ML.** PD, LGD, dan EWS belum dilatih; PD yang dipakai
berasal dari masukan analis di antarmuka. Ketika model asli siap, ia masuk
sebagai tool tambahan di `copilot/alat/` tanpa mengubah putaran agen.

---

## Profil model dan anggaran memori

Sasarannya Kaggle / Colab free tier: satu T4 16 GB dengan sebagian sudah dipakai
runtime notebook. Peran dipisah supaya model besar tidak ikut termuat saat tidak
dipakai.

| Profil | ekstraksi & chat | agen | embedding | Perkiraan |
|---|---|---|---|---|
| `hemat` (bawaan) | `qwen2.5:3b-instruct` | `qwen2.5:3b-instruct` | `nomic-embed-text` | ~2,3 GB |
| `seimbang` | SahabatAI (lihat di bawah) | `qwen2.5:7b-instruct` | `nomic-embed-text` | ~8 GB |

`hemat` jadi bawaan supaya repo ini bisa langsung dijalankan orang lain tanpa
mencari berkas GGUF dulu, dan tetap muat di CPU-only.

### Tentang tag SahabatAI

Profil `seimbang` menyerahkan peran bahasa Indonesia ke SahabatAI, tetapi tag
Ollama-nya **tidak dibawakan repo ini** dan harus Anda setel sendiri:

```bash
export COPILOT_PROFIL=seimbang
export COPILOT_MODEL_SAHABAT=hf.co/<pengguna>/<repo-gguf-sahabatai>:Q4_K_M
```

Alasannya, `ollama pull hf.co/...` hanya menerima repo yang berisi berkas GGUF,
sedangkan repo resmi SahabatAI (`GoToCompany/...`) diterbitkan dalam format
safetensors. Tag GGUF yang beredar adalah kuantisasi pihak ketiga yang datang
dan pergi, jadi menuliskan satu nama di sini hanya akan basi. Cari kuantisasi
GGUF dari `gemma2-9b-cpt-sahabatai-v1-instruct` (~6 GB pada Q4_K_M) atau
`llama3-8b-cpt-sahabatai-v1-instruct` (~5 GB) di Hugging Face, verifikasi
repo-nya, lalu isikan tagnya.

Tanpa variabel itu, profil `seimbang` berhenti dengan pesan yang menjelaskan
persis hal ini — bukan gagal di tengah demo.

### Menimpa per peran

`COPILOT_MODEL_*` menang atas profil:

```bash
export COPILOT_MODEL_AGEN=qwen2.5:7b-instruct   # agen saja yang dibesarkan
export COPILOT_ANGGARAN_GB=13                   # batas peringatan di sidebar
export COPILOT_MAKS_PUTARAN=8                   # batas putaran tool calling
```

---

## Menjalankan

### 1. Ollama dan model

```bash
bash copilot/scripts/siapkan_ollama.sh
```

Idempoten: memasang Ollama bila belum ada, menghidupkan servernya sebagai proses
latar (notebook tidak punya systemd), lalu menarik model sesuai profil.

### 2. Dependensi Python

```bash
pip install -r copilot/requirements.txt
```

Tipis dengan sengaja — tidak ada torch, transformers, FAISS, atau LangChain.
Inferensi dikerjakan Ollama di luar proses Python, dan pencarian atas korpus
seukuran ini selesai dengan satu perkalian matriks NumPy.

### 3. Index kebijakan

```bash
python -m copilot.rag.indeks
```

Memotong seluruh PDF di `docs/policies/` per pasal, meng-embed, lalu menyimpan
ke `data/index/`. Cukup sekali per korpus. POJK 40/2019 menghasilkan sekitar 105
potongan atas 77 pasal.

Artefaknya sengaja **tidak** masuk `data/gold/`: gold adalah warehouse keluaran
pipeline dan tunduk pada gerbang kualitasnya, sedangkan vektor tidak punya
hubungan apa pun dengan uji-uji itu.

Index menyimpan nama model embedding-nya. Bila konfigurasi berubah, pemuatan
menolak jalan dan meminta index dibangun ulang — vektor dari dua model berbeda
tidak sebanding, dan membiarkannya lewat menghasilkan sitasi yang terlihat wajar
tapi salah pasal.

### 4. Uji asap

```bash
python -m copilot.scripts.uji_asap
```

Memeriksa server, tiap model per peran, tool deterministik, embedding, index,
dan satu putaran agen penuh. Jalankan sebelum demo, bukan saat demo.

Yang paling sering ditangkapnya: model agen yang tidak mendukung function
calling. Gejalanya halus — model menjawab dengan kalimat berisi angka alih-alih
memanggil tool, dan tanpa uji ini kelihatan seperti berhasil.

### 5. Antarmuka

```bash
pip install -r app/ui/requirements.txt
streamlit run app/ui/app.py
```

Buka halaman **7 · Copilot lokal**.

---

## Catatan implementasi

**Ekstraksi dipotong per kelompok halaman (~6.000 karakter).** Model 3B efektif
hanya pada beberapa ribu karakter sekali baca meski jendela nominalnya jauh
lebih besar. Ini soal ketelitian, bukan soal muat. Satu potongan yang gagal
tidak membatalkan seluruh dokumen; kegagalannya tercatat di `catatan`.

**Penggabungan hasil dikerjakan Python, bukan model.** Meminta model meringkas
keluaran model hanya menambah kesempatan angka berubah tanpa jejak.

**Galat tool dikembalikan ke model, bukan dilempar ke atas.** Model yang
menerima pesan galat biasanya memperbaiki argumennya pada putaran berikutnya;
melempar exception memutus kesempatan itu dan membatalkan seluruh analisis.

**Batas putaran tidak membuang jejak.** Bila agen mencapai batasnya, angka yang
sudah dihitung tetap masuk memo, dan memo menyebutkan bahwa analisisnya tidak
ditutup.

**Argumen string dari model dirapikan seperlunya.** Titik itu ambigu:
`1.500.000.000` memakainya sebagai pemisah ribuan, `2.50` sebagai koma desimal.
Aturannya mengikuti jumlah tanda, bukan tebakan lokal. Kesalahan *nilai* —
satuan keliru, pos tertukar — sengaja dibiarkan lolos ke fungsi perhitungan
supaya tertangkap validasinya dan terlihat di jejak.

**Konstanta kebijakan di `alat/parameter.py` disalin, bukan diimpor, dari
`app/ui/lib/mock_engine.py`.** `app/ui/lib` adalah kode demo antarmuka yang akan
diganti pemanggilan FastAPI; lapisan tool tidak boleh ikut mati saat modul itu
dibongkar. Konsekuensinya disengaja: bila ambang kebijakan berubah, kedua berkas
harus diperbarui.

## Yang belum ditangani

- **PDF hasil pindaian.** Tidak ada OCR; dokumen tanpa lapisan teks ditolak
  dengan pesan yang menyebutkan sebabnya.
- **Tabel PDF berkolom rapat.** Ekstraksi teks pypdf mempertahankan baris tetapi
  tidak mempertahankan kolom. Rekening koran dengan tata letak padat bisa salah
  dipasangkan kolom debit/kreditnya, dan itu tidak selalu kelihatan dari
  hasilnya.
- **Multi-periode laporan keuangan.** Yang diambil satu periode terakhir.
- **Model PD/LGD/EWS**, sesuai lingkup tahap ini.
