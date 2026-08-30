"""Parameter kebijakan yang diturunkan dari `data/gold`, bukan ditulis tangan.

`lib/mock_engine.py` menyimpan sederet angka kebijakan sebagai konstanta: batas
BMPK, recovery per jenis agunan, ambang covenant per rating, dan matriks
kewenangan komite. Rumus yang memakainya memang harus tetap deterministik —
batas kredit adalah perhitungan aturan, bukan prediksi, dan komite tidak bisa
membela angka yang keluar dari regresi. Yang tidak bisa dibela adalah angkanya
sendiri dikarang padahal tabelnya sudah ada di lapisan emas.

Modul ini membaca angka itu dari tempat asalnya:

    batas dan eksposur BMPK   fact_eksposur_grup, dim_debitur.grup_id
    recovery per jenis agunan fact_agunan.haircut
    ambang covenant           fact_covenant x abt_pd.app_rating_internal
    matriks kewenangan        fact_pengajuan.komite_level x plafon_diminta_rp

Tiap keluaran membawa `sumber`. Ketika tabelnya tidak ada, konstanta lama tetap
dipakai sebagai cadangan dan sumbernya disebut "bawaan mesin demo" — supaya
layar bisa membedakan angka yang bisa ditelusuri ke tabel dari angka yang
sekadar diasumsikan.

Selisihnya bukan hal kecil. Batas BMPK pada mesin demo Rp 750 miliar, sedangkan
`fact_eksposur_grup` memakai Rp 3 triliun untuk seluruh grup; dan arah covenant
DER pada mesin demo justru terbalik dari data — di sana kelas rating lemah
mendapat ambang lebih longgar, karena covenant disetel relatif terhadap kondisi
debitur saat akad.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import mock_engine as me
from lib import model_nyata as mn

# Nama jenis agunan pada antarmuka vs slug pada `fact_agunan`. Dua jenis pada
# antarmuka tidak punya padanan di tabel: "Tanpa agunan" memang tidak beragunan,
# dan penjaminan korporasi tidak tercatat sebagai agunan berjaminan fisik.
# Keduanya tetap memakai angka kebijakan lama, dan itu ikut dilaporkan.
SLUG_AGUNAN = {
    "Deposito / cash collateral": "deposito",
    "Tanah dan bangunan pabrik (SHM/SHGB)": "tanah_bangunan",
    "Mesin dan peralatan": "mesin_peralatan",
    "Piutang dagang (fidusia)": "piutang_usaha",
    "Persediaan (fidusia)": "persediaan",
}

# Urutan tingkat kewenangan dari terendah. Dipakai untuk mengurutkan ambang yang
# ditemukan pada `fact_pengajuan`; namanya sendiri diambil apa adanya dari data.
URUTAN_KOMITE = ["Kepala Cabang", "Komite Kredit Wilayah", "Komite Kredit Pusat"]

# Ambang yang tidak ada di lapisan emas. DSCR tidak pernah tercatat sebagai
# jenis covenant pada `fact_covenant` (hanya DER, ICR, dan debt to EBITDA),
# jadi ia tetap kebijakan internal dan tidak boleh diakui sebagai turunan data.
JENIS_COVENANT_DATA = ("der_maks", "icr_min", "debt_to_ebitda_maks")


@st.cache_data(show_spinner=False)
def recovery_agunan() -> tuple[dict, dict]:
    """Recovery per jenis agunan dari `fact_agunan.haircut`.

    Haircut pada tabel adalah porsi nilai taksasi yang masih diakui — persis
    peran `RECOVERY_AGUNAN` pada mesin demo. Nilainya seragam per jenis
    (simpangan baku nol), jadi rata-ratanya bukan ringkasan yang mengaburkan
    apa pun.
    """
    ag = mn.gold("fact_agunan")
    if ag is None or "haircut" not in ag:
        return dict(me.RECOVERY_AGUNAN), {"sumber": "bawaan mesin demo", "tanpa_data": []}

    rata = ag.groupby("jenis")["haircut"].mean()
    nilai = dict(me.RECOVERY_AGUNAN)
    tanpa_data = []
    for nama, slug in SLUG_AGUNAN.items():
        if slug in rata.index:
            nilai[nama] = float(rata[slug])
        else:
            tanpa_data.append(nama)
    tanpa_data += [n for n in me.RECOVERY_AGUNAN if n not in SLUG_AGUNAN]
    return nilai, {"sumber": "fact_agunan.haircut", "tanpa_data": tanpa_data}


@st.cache_data(show_spinner=False)
def portofolio_berpita() -> pd.DataFrame | None:
    """Seluruh `abt_pd` diberi skor model PD lalu dipotong menjadi pita risiko.

    Ini tulang punggung seluruh grid di bawah. Sejak rating huruf tidak lagi
    dipakai antarmuka, kebijakan per kelas harus dikunci ke sesuatu yang benar
    dimiliki model — dan yang dimiliki model adalah pita pada
    `pd_decision_policy.json`. Portofolio pengembangan diskor sekali per proses,
    lalu tiap pita mewarisi kebijakan yang secara historis menyertainya.
    """
    bundel = mn.muat_pd()
    abt = mn.gold("abt_pd")
    if bundel is None or abt is None:
        return None
    data = abt.copy()
    if "fin_ebitda_nonpositif" not in data:
        data["fin_ebitda_nonpositif"] = (data["fin_ebitda_rp"] <= 0).astype(int)
    kurang = [f for f in bundel["features"] if f not in data.columns]
    if kurang:
        return None
    try:
        skor = bundel["model"].predict_proba(data[bundel["features"]])[:, 1]
    except Exception:
        return None
    data["skor_pd"] = skor
    data["pita"] = [mn._band(float(s), bundel["risk_cutoffs"]) for s in skor]
    return data


@st.cache_data(show_spinner=False)
def covenant_per_pita() -> tuple[dict, dict]:
    """Ambang covenant per pita risiko dari `fact_covenant`.

    Ambang dijoin ke pita lewat fasilitas dan pengajuannya. Hasilnya monoton:
    makin tinggi pita, makin longgar DER dan makin rendah ICR yang disyaratkan —
    covenant memang disetel relatif terhadap kondisi debitur saat akad, bukan
    sebagai hukuman.
    """
    cv = mn.gold("fact_covenant")
    fs = mn.gold("fact_fasilitas")
    data = portofolio_berpita()
    if cv is None or fs is None or data is None:
        return dict(me.COVENANT_PER_PITA), {"sumber": "bawaan mesin demo", "pita": []}

    gab = (
        cv[["facility_id", "jenis", "ambang"]]
        .merge(fs[["facility_id", "application_id"]], on="facility_id")
        .merge(data[["application_id", "pita"]], on="application_id")
    )
    grid = gab.groupby(["pita", "jenis"])["ambang"].median().unstack()

    keluar = {}
    terisi = []
    for pita, bawaan in me.COVENANT_PER_PITA.items():
        baris = dict(bawaan)
        if pita in grid.index:
            for jenis in JENIS_COVENANT_DATA:
                if jenis in grid.columns and pd.notna(grid.loc[pita, jenis]):
                    baris[jenis] = float(grid.loc[pita, jenis])
            terisi.append(pita)
        keluar[pita] = baris
    return keluar, {
        "sumber": "fact_covenant x pita model PD",
        "pita": terisi,
        "tetap_internal": ["dscr_min", "uji"],
    }


@st.cache_data(show_spinner=False)
def pricing_per_pita() -> tuple[dict, dict]:
    """Suku bunga dasar per pita, dari pricing yang benar-benar pernah ditagih.

    Menggantikan tumpukan biaya dana + operasional + margin + expected loss yang
    ditulis tangan pada mesin demo. Tumpukan itu memberi harga 116 sampai 618
    bps di atas praktik buku ini, dan selisihnya melebar justru pada pita
    berisiko - persis kasus yang paling sensitif dinegosiasikan.

    Perhatikan apa yang TIDAK ditambahkan di sini: premi expected loss per
    pengajuan. Skor PD artefak ini tidak terkalibrasi, jadi PD x LGD x EAD tidak
    bisa dibaca sebagai rupiah kerugian, dan menambahkannya ke pricing berarti
    menambahkan angka yang tidak punya satuan.
    """
    pg = mn.gold("fact_pengajuan")
    data = portofolio_berpita()
    if pg is None or data is None or "pricing_bps" not in pg:
        return dict(me.PRICING_PER_PITA), {"sumber": "bawaan mesin demo", "pita": []}

    gab = data[["application_id", "pita"]].merge(
        pg[["application_id", "pricing_bps"]], on="application_id")
    median = gab.groupby("pita")["pricing_bps"].median()
    keluar = dict(me.PRICING_PER_PITA)
    terisi = []
    for pita in me.PRICING_PER_PITA:
        if pita in median.index:
            keluar[pita] = float(median[pita]) / 1e4
            terisi.append(pita)
    return keluar, {"sumber": "fact_pengajuan.pricing_bps per pita", "pita": terisi}


@st.cache_data(show_spinner=False)
def pagu_per_pita() -> tuple[dict, dict]:
    """Pagu limit per pita, dari plafon yang pernah benar-benar disetujui.

    Persentil 95 plafon disetujui pada tiap pita — bukan angka kewenangan resmi,
    melainkan langit-langit yang terbaca dari praktik. Disebut apa adanya di
    layar, karena bedanya tipis antar pita dan karena itu jarang mengikat.
    """
    data = portofolio_berpita()
    if data is None or "app_plafon_rp" not in data:
        return dict(me.PAGU_PER_PITA), {"sumber": "bawaan mesin demo", "pita": []}
    p95 = data.groupby("pita")["app_plafon_rp"].quantile(0.95)
    keluar = dict(me.PAGU_PER_PITA)
    terisi = []
    for pita in me.PAGU_PER_PITA:
        if pita in p95.index and pd.notna(p95[pita]):
            keluar[pita] = float(round(p95[pita] / 1e9) * 1e9)
            terisi.append(pita)
    return keluar, {"sumber": "abt_pd.app_plafon_rp persentil 95 per pita", "pita": terisi}


@st.cache_data(show_spinner=False)
def matriks_kewenangan() -> tuple[list, dict]:
    """Ambang kewenangan komite dari keputusan yang sudah pernah diambil.

    Tiap tingkat menangani rentang plafon yang tidak bertumpuk pada data, jadi
    ambangnya dibaca sebagai plafon terbesar yang pernah diputus tingkat itu.
    Tingkat teratas tidak diberi ambang atas.
    """
    pg = mn.gold("fact_pengajuan")
    if pg is None or "komite_level" not in pg:
        return list(me.MATRIKS_KEWENANGAN), {"sumber": "bawaan mesin demo"}

    batas_atas = pg.groupby("komite_level")["plafon_diminta_rp"].max()
    tingkat = [k for k in URUTAN_KOMITE if k in batas_atas.index]
    tingkat += [k for k in batas_atas.index if k not in URUTAN_KOMITE]
    if not tingkat:
        return list(me.MATRIKS_KEWENANGAN), {"sumber": "bawaan mesin demo"}

    # Dibulatkan ke miliar terdekat. Plafon terbesar yang tercatat pada satu
    # tingkat adalah Rp 24,997 M; ambang yang dipakai apa adanya akan menaikkan
    # pengajuan Rp 25 M ke tingkat di atasnya hanya karena pembulatan data.
    matriks = [(round(float(batas_atas[k]) / 1e9) * 1e9, k) for k in tingkat[:-1]]
    matriks.append((float("inf"), tingkat[-1]))
    return matriks, {
        "sumber": "fact_pengajuan.komite_level",
        "tingkat": tingkat,
    }


# --------------------------------------------------------------------------
# Nilai rujukan untuk fitur yang tidak ada pada berkas pengajuan
# --------------------------------------------------------------------------
# Tingkat pemakaian plafon, konversi EBITDA ke kas, konsentrasi mitra, dan
# porsi utang berbunga tidak tertulis di dokumen mana pun — mereka datang dari
# warehouse dan lapisan graf. Selama ini `lengkapi_fitur_graf()` mengisinya
# dengan angka yang ditulis tangan sebagai "khas segmen". Beberapa di antaranya
# meleset jauh dari portofolio yang sebenarnya, dan satu meleset ke arah yang
# berbahaya: porsi utang berbunga ditaksir 0,50 padahal terukur 0,88, sehingga
# ICR dan DSCR pemohon terbaca lebih sehat daripada seharusnya.
ASUMSI_BAWAAN = {
    "konversi_ebitda_kas": 0.74,
    "utilisasi_plafon": 0.74,
    "porsi_utang_berbunga": 0.88,
    "buyer_concentration_hhi": 0.96,
    "supplier_concentration_hhi": 0.95,
    "neighbor_default_rate_1hop": 0.0,
    "group_exposure_share": 0.01,
}


@st.cache_data(show_spinner=False)
def asumsi_portofolio() -> tuple[dict, dict]:
    """Median portofolio untuk tiap fitur yang tidak bisa dibaca dari berkas.

    Median, bukan rata-rata: sebaran rasio keuangan berekor panjang, dan satu
    debitur dengan EBITDA nyaris nol sudah cukup menggeser rataan. Tiap nilai
    dilaporkan bersama tabel asalnya supaya halaman bisa menyebut "median
    portofolio" alih-alih membiarkannya terbaca sebagai pengukuran atas pemohon.
    """
    nilai = dict(ASUMSI_BAWAAN)
    asal: dict[str, str] = {k: "bawaan modul" for k in nilai}

    abt = mn.gold("abt_pd")
    if abt is not None and "fin_cfo_to_ebitda" in abt:
        cfo = pd.to_numeric(abt["fin_cfo_to_ebitda"], errors="coerce")
        # Rasio negatif berarti arus kas operasi negatif, dan di atas tiga
        # berarti EBITDA nyaris nol. Keduanya nyata tetapi bukan titik tengah.
        cfo = cfo[(cfo > 0) & (cfo < 3)]
        if len(cfo) > 100:
            nilai["konversi_ebitda_kas"] = float(round(cfo.median(), 3))
            asal["konversi_ebitda_kas"] = "abt_pd.fin_cfo_to_ebitda (median)"
        d = abt[["fin_debt_to_ebitda", "fin_ebitda_rp", "fin_total_liabilitas_rp"]].apply(
            pd.to_numeric, errors="coerce")
        d = d[(d["fin_ebitda_rp"] > 0) & (d["fin_total_liabilitas_rp"] > 0)
              & d["fin_debt_to_ebitda"].between(0, 20)]
        porsi = (d["fin_debt_to_ebitda"] * d["fin_ebitda_rp"]) / d["fin_total_liabilitas_rp"]
        porsi = porsi[porsi.between(0, 2)]
        if len(porsi) > 100:
            nilai["porsi_utang_berbunga"] = float(round(porsi.median(), 3))
            asal["porsi_utang_berbunga"] = "abt_pd: debt/EBITDA x EBITDA / liabilitas (median)"

    fas = mn.gold("fact_fasilitas")
    if fas is not None and "pemakaian_plafon_pct" in fas:
        util = pd.to_numeric(fas["pemakaian_plafon_pct"], errors="coerce").dropna()
        if len(util) > 100:
            nilai["utilisasi_plafon"] = float(round(util.median(), 3))
            asal["utilisasi_plafon"] = "fact_fasilitas.pemakaian_plafon_pct (median)"

    pit = mn.gold("feat_graf_pit")
    if pit is not None:
        for kolom in ("buyer_concentration_hhi", "supplier_concentration_hhi",
                      "neighbor_default_rate_1hop"):
            if kolom not in pit:
                continue
            seri = pd.to_numeric(pit[kolom], errors="coerce").dropna()
            if len(seri) > 100:
                nilai[kolom] = float(round(seri.median(), 4))
                asal[kolom] = f"feat_graf_pit.{kolom} (median)"

    eg = mn.gold("fact_eksposur_grup")
    if eg is not None and "group_exposure_share" in eg:
        seri = pd.to_numeric(eg["group_exposure_share"], errors="coerce").dropna()
        if len(seri) > 100:
            nilai["group_exposure_share"] = float(round(seri.median(), 4))
            asal["group_exposure_share"] = "fact_eksposur_grup.group_exposure_share (median)"

    return nilai, asal


@st.cache_data(show_spinner=False)
def utang_berbunga(cif: tuple[int, ...]) -> dict | None:
    """Baki debet fasilitas yang tercatat atas debitur tercocok.

    Dipakai menggantikan taksiran "sekian persen dari total liabilitas" begitu
    pemohon berhasil dicocokkan ke debitur eksisting. Tanpa cocokan hasilnya
    `None`, dan pemanggilnya kembali ke taksiran — yang harus disebut sebagai
    taksiran.
    """
    if not cif:
        return None
    fas = mn.gold("fact_fasilitas")
    if fas is None or "outstanding_rp" not in fas:
        return None
    milik = fas[fas["cif_sk"].isin(cif)]
    if milik.empty:
        return None
    return {
        "nilai": float(pd.to_numeric(milik["outstanding_rp"], errors="coerce").sum()),
        "jumlah_fasilitas": int(len(milik)),
        "sumber": "fact_fasilitas.outstanding_rp",
    }


@st.cache_data(show_spinner=False)
def _peta_grup() -> pd.DataFrame | None:
    deb = mn.gold("dim_debitur")
    if deb is None or "grup_id" not in deb:
        return None
    if "is_current" in deb:
        deb = deb[deb["is_current"].astype(bool)]
    return deb[["cif_sk", "grup_id"]].dropna().drop_duplicates()


@st.cache_data(show_spinner=False)
def eksposur_grup(cif: tuple[int, ...], tanggal: pd.Timestamp) -> dict | None:
    """Eksposur dan sisa ruang BMPK grup, dari snapshot terakhir sebelum `tanggal`.

    `cif` adalah debitur eksisting yang tercocok dari dokumen pemohon. Tanpa satu
    pun cocokan, tidak ada grup yang bisa ditunjuk dan fungsi ini mengembalikan
    `None` — pemanggilnya wajib menampilkan itu sebagai "tidak terukur", bukan
    menggantinya dengan porsi asumsi.

    Bila cocokan tersebar di beberapa grup, yang dilaporkan adalah grup dengan
    porsi terpakai terbesar. Grup lain disebut pada catatan: menjumlahkan
    eksposur lintas grup akan salah, karena BMPK dihitung per grup debitur.
    """
    if not cif:
        return None
    peta = _peta_grup()
    eg = mn.gold("fact_eksposur_grup")
    if peta is None or eg is None:
        return None

    grup = peta[peta["cif_sk"].isin(cif)]["grup_id"].unique()
    if len(grup) == 0:
        return None

    eg = eg[eg["grup_id"].isin(grup)].copy()
    eg["snapshot_date"] = pd.to_datetime(eg["snapshot_date"])
    eg = eg[eg["snapshot_date"] <= pd.Timestamp(tanggal)]
    if eg.empty:
        return None

    terakhir = (
        eg.sort_values("snapshot_date").groupby("grup_id", as_index=False).last()
        .sort_values("group_exposure_share", ascending=False)
    )
    utama = terakhir.iloc[0]
    catatan = []
    if len(terakhir) > 1:
        lain = ", ".join(str(int(g)) for g in terakhir["grup_id"].iloc[1:])
        catatan.append(
            f"Cocokan menyentuh {len(terakhir)} grup debitur ({lain} selain grup "
            f"{int(utama['grup_id'])}); yang dilaporkan grup dengan porsi terpakai "
            "terbesar, karena BMPK dihitung per grup dan tidak boleh dijumlahkan."
        )
    return {
        "grup_id": int(utama["grup_id"]),
        "jumlah_grup": int(len(terakhir)),
        "eksposur_rp": float(utama["total_eksposur_rp"]),
        "batas_bmpk_rp": float(utama["batas_bmpk_rp"]),
        "share": float(utama["group_exposure_share"]),
        "sisa_ruang_rp": float(utama["sisa_ruang_rp"]),
        "snapshot": pd.Timestamp(utama["snapshot_date"]),
        "sumber": "fact_eksposur_grup",
        "catatan": catatan,
    }


def terapkan() -> dict:
    """Pasang parameter hasil kalibrasi ke `mock_engine`, lalu laporkan asalnya.

    Dipanggil sekali di awal tiap halaman. Mesin skoring membaca nilainya lewat
    `mock_engine.KALIBRASI`, bukan lewat impor balik ke modul ini — arah
    ketergantungannya tetap satu arah dan mesin tetap bisa dipakai tanpa data
    emas sama sekali.
    """
    recovery, asal_recovery = recovery_agunan()
    covenant, asal_covenant = covenant_per_pita()
    kewenangan, asal_kewenangan = matriks_kewenangan()
    pricing, asal_pricing = pricing_per_pita()
    pagu, asal_pagu = pagu_per_pita()

    # Ambang pita ikut dititipkan supaya `mock_engine` bisa memotong skor
    # menjadi pita tanpa mengimpor lapisan model — arah ketergantungannya tetap
    # satu arah, dan mesin tetap jalan tanpa artefak sama sekali.
    bundel = mn.muat_pd()
    if bundel and bundel.get("risk_cutoffs"):
        me.KALIBRASI["cutoffs_pd"] = dict(bundel["risk_cutoffs"])

    me.KALIBRASI.update(
        recovery_agunan=recovery,
        covenant_per_pita=covenant,
        matriks_kewenangan=kewenangan,
        pricing_per_pita=pricing,
        pagu_per_pita=pagu,
    )
    return {
        "recovery": asal_recovery,
        "covenant": asal_covenant,
        "kewenangan": asal_kewenangan,
        "pricing": asal_pricing,
        "pagu": asal_pagu,
    }


def ringkas_sumber(laporan: dict) -> list[str]:
    """Baris pendek untuk sidebar: parameter mana yang sudah turun dari data."""
    return [
        f"Recovery agunan · {laporan['recovery']['sumber']}",
        f"Ambang covenant · {laporan['covenant']['sumber']}",
        f"Suku bunga dasar · {laporan['pricing']['sumber']}",
        f"Pagu limit · {laporan['pagu']['sumber']}",
        f"Matriks kewenangan · {laporan['kewenangan']['sumber']}",
    ]
