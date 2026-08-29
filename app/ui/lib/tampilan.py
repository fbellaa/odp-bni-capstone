"""Komponen tampilan yang dipakai bersama oleh seluruh halaman.

Palet dan komponen di sini mengikuti kosakata segmen komersial: badan hukum,
grup usaha, pemilik manfaat, counterparty dagang, dan agunan.
"""
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import mock_engine
from lib.format import kali, miliar, persen

JUDUL_APLIKASI = "Agentic AI Copilot untuk Keputusan Kredit Komersial"
SUBJUDUL_APLIKASI = (
    "Segmen komersial — debitur menengah dengan penjualan tahunan Rp 30 sampai 300 miliar."
)

# --------------------------------------------------------------------------
# Palet aplikasi
# --------------------------------------------------------------------------
# Biru laut sebagai warna induk, teal sebagai aksen, dan deret amber-koral-merah
# untuk tingkat risiko. Ketiganya dipilih pada tingkat kecerahan yang mirip
# supaya tidak ada satu warna pun yang "berteriak" di layar ruang komite.
PRIMER = "#1d5fa8"
PRIMER_TUA = "#123a63"
AKSEN = "#12a594"
AMBER = "#e0a02a"
KORAL = "#e2683f"
MERAH = "#cc3b52"
UNGU = "#7b5ea7"
HIJAU = "#1f8a5f"
ABU = "#8b97a6"

DERET_RISIKO = [HIJAU, AKSEN, AMBER, KORAL, MERAH]
DERET_KATEGORI = [PRIMER, AKSEN, AMBER, UNGU, KORAL, "#3f8fa8", HIJAU, "#a4553a"]

GAYA = """
<style>
  :root {
    --primer:#1d5fa8; --primer-tua:#123a63; --aksen:#12a594;
    --amber:#e0a02a; --koral:#e2683f; --merah:#cc3b52;
    --garis:rgba(29,95,168,.16); --lembut:rgba(29,95,168,.06);
  }
  .block-container {padding-top:1.6rem; padding-bottom:3.5rem; max-width:1400px;}

  /* Kepala halaman */
  .hero {display:flex; gap:1.1rem; align-items:flex-start;
         background:linear-gradient(115deg,var(--primer-tua) 0%,var(--primer) 52%,#1b7f9b 100%);
         color:#fff; border-radius:18px; padding:1.15rem 1.4rem; margin-bottom:1.1rem;
         box-shadow:0 10px 26px -16px rgba(18,58,99,.85);}
  .hero-nomor {font-size:2.1rem; font-weight:800; line-height:1;
               background:rgba(255,255,255,.16); border-radius:14px;
               padding:.55rem .85rem; min-width:3.1rem; text-align:center;}
  .hero-teks h1 {font-size:1.5rem; margin:0 0 .2rem 0; color:#fff; font-weight:700;
                 letter-spacing:-.01em; padding:0;}
  .hero-teks p {margin:0; font-size:.92rem; opacity:.9; line-height:1.45;}
  .hero-chips {margin-top:.6rem; display:flex; flex-wrap:wrap; gap:.4rem;}
  .hero-chip {background:rgba(255,255,255,.15); border:1px solid rgba(255,255,255,.28);
              border-radius:999px; padding:.16rem .65rem; font-size:.75rem;}
  .hero-chip b {font-weight:700;}

  /* Judul bagian */
  .bagian {margin:1.5rem 0 .55rem 0; border-left:4px solid var(--aksen); padding-left:.7rem;}
  .bagian h3 {margin:0; font-size:1.06rem; font-weight:700; letter-spacing:-.01em;}
  .bagian p {margin:.15rem 0 0 0; font-size:.83rem; opacity:.7;}

  /* Metrik */
  div[data-testid="stMetric"] {background:#fff; border:1px solid var(--garis);
        border-radius:14px; padding:.75rem .9rem;
        box-shadow:0 2px 10px -8px rgba(18,58,99,.55);}
  div[data-testid="stMetricValue"] {font-size:1.4rem; font-weight:700;}
  div[data-testid="stMetricLabel"] p {font-size:.78rem; opacity:.72; font-weight:600;}

  /* Kartu */
  .badge {display:inline-block; padding:.2rem .65rem; border-radius:999px;
          font-size:.74rem; font-weight:700; letter-spacing:.02em;}
  .kotak {background:#fff; border:1px solid var(--garis); border-radius:12px;
          padding:.85rem 1rem; margin-bottom:.6rem;}
  .kartu {background:#fff; border:1px solid var(--garis); border-left:5px solid var(--w);
          border-radius:12px; padding:.75rem .95rem; margin-bottom:.55rem;
          box-shadow:0 2px 10px -9px rgba(18,58,99,.6);}
  .kartu-judul {font-weight:700; font-size:.92rem; color:var(--w);}
  .kartu-isi {font-size:.85rem; opacity:.82; margin-top:.15rem; line-height:1.5;}
  .gerbang {background:#fff; border:1px solid var(--garis); border-left:5px solid var(--w);
            border-radius:12px; padding:.75rem .95rem; margin-bottom:.55rem;}
  .gerbang .aspek {font-weight:700; letter-spacing:.05em; text-transform:uppercase;
            font-size:.7rem;}
  .gerbang .pasal {font-size:.75rem; opacity:.6; font-family:ui-monospace,monospace;}
  .tipis {opacity:.72; font-size:.85rem;}

  /* Tab */
  button[data-baseweb="tab"] {font-weight:600;}
  div[data-baseweb="tab-list"] {gap:.35rem; border-bottom:1px solid var(--garis);}
  button[data-baseweb="tab"][aria-selected="true"] {background:var(--lembut);
        border-radius:10px 10px 0 0;}

  /* Sidebar */
  section[data-testid="stSidebar"] {background:#f0f4fa; border-right:1px solid var(--garis);}
  .sidebar-merek {font-size:1.02rem; line-height:1.25; color:var(--primer-tua);
        border-bottom:1px solid var(--garis); padding-bottom:.6rem; margin-bottom:.7rem;}
  .sidebar-merek b {font-size:1.12rem;}

  /* Tombol utama */
  button[kind="primary"] {border-radius:10px; font-weight:700;
        box-shadow:0 6px 16px -10px rgba(29,95,168,.9);}

  /* Tabel dan expander */
  div[data-testid="stDataFrame"] {border-radius:12px; overflow:hidden;
        border:1px solid var(--garis);}
  details[data-testid="stExpander"] {border-radius:12px; border:1px solid var(--garis);}
</style>
"""


def gaya_plot(fig, tinggi: int | None = None):
    """Satu selera untuk semua grafik: latar bersih, kisi tipis, huruf sama."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Source Sans Pro, sans-serif", size=12, color="#16212e"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(29,95,168,.03)",
        margin=dict(l=10, r=10, t=44, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="rgba(29,95,168,.3)"),
    )
    fig.update_xaxes(gridcolor="rgba(29,95,168,.10)", zerolinecolor="rgba(29,95,168,.22)")
    fig.update_yaxes(gridcolor="rgba(29,95,168,.10)", zerolinecolor="rgba(29,95,168,.22)")
    if tinggi:
        fig.update_layout(height=tinggi)
    return fig


PALET_TIPE = {
    "Badan hukum": PRIMER,
    "Grup usaha": PRIMER_TUA,
    "Pemilik manfaat": UNGU,
    "Pengurus": "#a35b86",
    "Counterparty": AKSEN,
    "Atribut berbagi": MERAH,
    "Agunan": AMBER,
}

SIMBOL_TIPE = {
    "Badan hukum": "circle",
    "Grup usaha": "hexagon",
    "Pemilik manfaat": "diamond",
    "Pengurus": "diamond-open",
    "Counterparty": "square",
    "Atribut berbagi": "x",
    "Agunan": "triangle-up",
}

PALET_KOMUNITAS = [
    PRIMER, AMBER, AKSEN, MERAH, UNGU, "#8a7f2e",
    "#3f8fa8", KORAL, "#5f7d3a", "#a35b86", "#3b6b8f", "#9c6b1f",
]

WARNA_STATUS = {
    mock_engine.LOLOS: HIJAU,
    mock_engine.TELAAH: AMBER,
    mock_engine.PENYESUAIAN: MERAH,
}

WARNA_KEPUTUSAN = {
    "SETUJU": HIJAU,
    "SETUJU DENGAN SYARAT": AMBER,
    "PERLU PENYESUAIAN": KORAL,
    "TOLAK": MERAH,
}


def setup_halaman(judul: str, ikon: str = "•") -> None:
    st.set_page_config(page_title=f"{judul} · Commercial Credit Copilot", page_icon=ikon, layout="wide")
    st.markdown(GAYA, unsafe_allow_html=True)


def hero(nomor: str, judul: str, ringkas: str,
         sorotan: list[tuple[str, str]] | None = None) -> None:
    """Kepala halaman: satu blok warna yang menyatakan halaman ini soal apa.

    Judul tidak lagi memakai `st.title` supaya tiap halaman dibuka dengan bidang
    warna berbentuk sama — layar ruang komite dibaca dari jauh, dan blok berwarna
    lebih cepat dikenali daripada teks polos.
    """
    isi = "".join(
        f'<span class="hero-chip"><b>{nilai}</b> {label}</span>'
        for label, nilai in (sorotan or [])
    )
    st.markdown(
        f'<div class="hero"><div class="hero-nomor">{nomor}</div>'
        f'<div class="hero-teks"><h1>{judul}</h1><p>{ringkas}</p>'
        f'<div class="hero-chips">{isi}</div></div></div>',
        unsafe_allow_html=True,
    )


def judul_bagian(teks: str, keterangan: str = "") -> None:
    st.markdown(
        f'<div class="bagian"><h3>{teks}</h3>'
        + (f"<p>{keterangan}</p>" if keterangan else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def kartu(judul: str, isi: str, warna: str = AKSEN, ikon: str = "") -> str:
    """Kartu berpita warna untuk temuan, pola, dan catatan pendek."""
    return (
        f'<div class="kartu" style="--w:{warna}">'
        f'<div class="kartu-judul">{ikon} {judul}</div>'
        f'<div class="kartu-isi">{isi}</div></div>'
    )


def sidebar_status() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-merek">Commercial<br><b>Credit Copilot</b></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Batas segmen: penjualan Rp {mock_engine.SEGMEN['penjualan_min'] / 1e9:.0f}–"
            f"{mock_engine.SEGMEN['penjualan_maks'] / 1e9:.0f} M · plafon Rp "
            f"{mock_engine.SEGMEN['plafon_min'] / 1e9:.0f}–"
            f"{mock_engine.SEGMEN['plafon_maks'] / 1e9:.0f} M"
        )
        st.caption("Status: under development · Agustus 2026")


def badge(teks: str, warna: str) -> str:
    return (
        f'<span class="badge" style="background:{warna}22;color:{warna};'
        f'border:1px solid {warna}55">{teks}</span>'
    )


def badge_keputusan(keputusan: str) -> str:
    return badge(keputusan, WARNA_KEPUTUSAN.get(keputusan, "#666666"))


def badge_grade(grade: str) -> str:
    return badge(f"Rating {grade}", mock_engine.WARNA_GRADE.get(grade, "#666666"))


def kartu_hasil(hasil: mock_engine.HasilSkor, plafon_diminta: float) -> None:
    """Baris metrik inti keputusan pada segmen komersial."""
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Probability of default", persen(hasil.pd), help="PD terkalibrasi 12 bulan ke depan")
    k2.metric("Rating internal", hasil.grade,
              help="Varian scorecard WOE menjadi rating yang mudah diaudit komite")
    k3.metric("Expected loss", miliar(hasil.expected_loss, 2), help="PD × LGD × EAD")
    k4.metric(
        "Usulan limit grup",
        miliar(hasil.limit_usulan, 0),
        delta=None if hasil.limit_usulan >= plafon_diminta
        else f"-{miliar(plafon_diminta - hasil.limit_usulan, 0)} dari permintaan",
        delta_color="inverse",
    )
    k5.metric("Usulan pricing", persen(hasil.pricing),
              help="Biaya dana + operasional + margin target + expected loss")


def kartu_rasio(hasil: mock_engine.HasilSkor) -> None:
    """Rasio keuangan komersial yang menjadi dasar covenant."""
    cov = hasil.covenant
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Interest coverage", kali(hasil.icr),
              delta=f"min {kali(cov['icr_min'])}",
              delta_color="normal" if hasil.icr >= cov["icr_min"] else "inverse")
    r2.metric("Debt to EBITDA", kali(hasil.debt_to_ebitda))
    r3.metric("Debt to equity", kali(hasil.der),
              delta=f"maks {kali(cov['der_maks'])}",
              delta_color="normal" if hasil.der <= cov["der_maks"] else "inverse")
    r4.metric("Debt service coverage", kali(hasil.dscr),
              delta=f"min {kali(cov['dscr_min'])}",
              delta_color="normal" if hasil.dscr >= cov["dscr_min"] else "inverse")
    r5.metric("Pertanggungan agunan", persen(hasil.coverage_agunan, 0),
              delta=f"min {persen(mock_engine.COVERAGE_MIN[hasil.grade], 0)}",
              delta_color="normal"
              if hasil.coverage_agunan >= mock_engine.COVERAGE_MIN[hasil.grade] else "inverse")


def panel_gerbang(gerbang: list[dict], ringkas: bool = False) -> None:
    """Gerbang kepatuhan pada alur keputusan (proposal 5.3).

    Setiap butir selalu disertai pasal sebagai dasar; tanpa kutipan, tidak ada
    jawaban.
    """
    for a in gerbang:
        warna = WARNA_STATUS.get(a["status"], "#666666")
        isi = (
            f'<div class="gerbang" style="--w:{warna}">'
            f'<span class="aspek" style="color:{warna}">{a["aspek"]} · {a["status"]}</span><br>'
            f'<b>{a["temuan"]}</b><br>'
        )
        if not ringkas:
            isi += f'<span class="tipis">{a["kutipan"]}</span><br>'
        isi += (
            f'<span class="pasal">{a["pasal"]}</span><br>'
            f'<span class="tipis">Tindakan: {a["tindakan"]}</span></div>'
        )
        st.markdown(isi, unsafe_allow_html=True)


def plot_bmpk(eksposur_grup: float, tambahan: float = 0.0,
              batas: float | None = None) -> go.Figure:
    """Posisi eksposur grup terhadap batas maksimum pemberian kredit."""
    batas = batas if batas is not None else mock_engine.BATAS_BMPK_GRUP
    sisa = max(batas - eksposur_grup - tambahan, 0.0)
    fig = go.Figure()
    potongan = [
        ("Eksposur grup berjalan", eksposur_grup, PRIMER),
        ("Usulan fasilitas ini", tambahan, AMBER),
        ("Sisa ruang BMPK", sisa, "rgba(139,151,166,.28)"),
    ]
    for nama, nilai, warna in potongan:
        fig.add_trace(go.Bar(
            x=[nilai / 1e9], y=["BMPK grup"], name=nama, orientation="h",
            marker_color=warna,
            hovertemplate=f"<b>{nama}</b><br>Rp %{{x:.1f}} M<extra></extra>",
        ))
    fig.add_vline(x=batas / 1e9, line_dash="dash", line_color=MERAH,
                  annotation_text=f"Batas Rp {batas / 1e9:.0f} M", annotation_position="top right")
    fig.update_layout(barmode="stack", xaxis_title="Rp miliar", yaxis_title=None)
    return gaya_plot(fig, 200)


def plot_kontribusi(kontribusi, jumlah: int = 8) -> go.Figure:
    """Bar horizontal ala SHAP untuk reason code."""
    data = list(kontribusi)[:jumlah][::-1]
    fig = go.Figure(
        go.Bar(
            x=[k.dampak for k in data],
            y=[k.fitur for k in data],
            orientation="h",
            marker_color=[MERAH if k.dampak > 0 else AKSEN for k in data],
            customdata=[k.nilai for k in data],
            hovertemplate="<b>%{y}</b><br>Nilai: %{customdata}<br>Dampak: %{x:+.3f} log-odds<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Dampak terhadap risiko (log-odds)", yaxis_title=None, showlegend=False,
    )
    return gaya_plot(fig, 70 + 34 * len(data))


def plot_kepemilikan(rantai: pd.DataFrame, entity_id: str) -> go.Figure:
    """Sankey penelusuran kepemilikan berlapis sampai pemilik manfaat akhir."""
    label = [entity_id]
    for _, r in rantai.iterrows():
        if r["pemilik"] not in label:
            label.append(r["pemilik"])
    indeks = {n: i for i, n in enumerate(label)}

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=18, thickness=16,
            line=dict(color="rgba(120,120,120,.5)", width=0.8),
            label=label,
            color=[PRIMER] + [
                UNGU if "Pemilik manfaat" in j else "#3b6b8f"
                for j in rantai["jenis"]
            ],
        ),
        link=dict(
            source=[indeks[r["pemilik"]] for _, r in rantai.iterrows()],
            target=[indeks[r["dimiliki"]] for _, r in rantai.iterrows()],
            value=[float(r["porsi_langsung"]) * 100 for _, r in rantai.iterrows()],
            color="rgba(47,111,159,.32)",
            customdata=[
                [f"{r['porsi_langsung'] * 100:.1f}%", f"{r['porsi_efektif'] * 100:.1f}%", r["jenis"]]
                for _, r in rantai.iterrows()
            ],
            hovertemplate="Kepemilikan langsung %{customdata[0]}<br>"
                          "Kepemilikan efektif %{customdata[1]}<br>"
                          "%{customdata[2]}<extra></extra>",
        ),
    ))
    fig.update_layout(font_size=11)
    return gaya_plot(fig, 320)


def _tata_letak_graf(nodes: pd.DataFrame, edges: pd.DataFrame):
    """Spring layout sederhana (Fruchterman-Reingold ringkas) tanpa dependensi tambahan."""
    try:
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from(nodes["id"])
        if not edges.empty:
            g.add_edges_from(zip(edges["source"], edges["target"]))
        return nx.spring_layout(g, seed=7, k=1.25 / math.sqrt(max(len(g), 1)))
    except Exception:
        # Cadangan: tata letak radial per hop bila networkx tidak terpasang.
        posisi = {}
        for hop, kelompok in nodes.groupby("hop"):
            n = len(kelompok)
            for i, nid in enumerate(kelompok["id"]):
                sudut = 2 * math.pi * i / max(n, 1)
                r = float(hop)
                posisi[nid] = (r * math.cos(sudut), r * math.sin(sudut))
        return posisi


# Relasi yang layak diberi penekanan visual karena menjadi dasar penggabungan
# grup debitur dan pemeriksaan BMPK.
RELASI_STRUKTURAL = {"memiliki", "mengendalikan", "menjabat_di", "menjamin_silang", "berbagi_atribut"}


def plot_graf(nodes: pd.DataFrame, edges: pd.DataFrame, warnai: str = "tipe",
              sorot: str | None = None) -> go.Figure:
    """Graf simpul-tepi dengan Plotly (lihat proposal 9.4).

    Relasi struktural (kepemilikan, pengendalian, rangkap jabatan, penjaminan
    silang, atribut berbagi) digambar lebih tebal daripada relasi dagang supaya
    tulang punggung grup usaha langsung terbaca.
    """
    posisi = _tata_letak_graf(nodes, edges)
    jejak = []

    for struktural in (False, True):
        gx, gy = [], []
        for _, e in edges.iterrows():
            if (e["relasi"] in RELASI_STRUKTURAL) != struktural:
                continue
            if e["source"] not in posisi or e["target"] not in posisi:
                continue
            x0, y0 = posisi[e["source"]]
            x1, y1 = posisi[e["target"]]
            gx += [x0, x1, None]
            gy += [y0, y1, None]
        if not gx:
            continue
        jejak.append(go.Scatter(
            x=gx, y=gy, mode="lines", hoverinfo="skip",
            line=dict(
                width=2.0 if struktural else 0.9,
                color="rgba(47,111,159,.55)" if struktural else "rgba(130,130,130,.35)",
            ),
        ))

    if warnai == "komunitas":
        warna = [PALET_KOMUNITAS[int(c) % len(PALET_KOMUNITAS)] for c in nodes["community_id"]]
    else:
        warna = [PALET_TIPE.get(t, "#888888") for t in nodes["tipe"]]

    ukuran = [30 if nid == sorot else (18 if h <= 1 else 12)
              for nid, h in zip(nodes["id"], nodes["hop"])]
    garis_tepi = ["#111111" if nid == sorot else "rgba(255,255,255,.65)" for nid in nodes["id"]]
    simbol = [SIMBOL_TIPE.get(t, "circle") for t in nodes["tipe"]]

    teks = []
    for r in nodes.itertuples():
        potong = [f"<b>{r.id}</b>", f"Tipe: {r.tipe}"]
        peran = getattr(r, "peran", None)
        if isinstance(peran, str) and peran not in ("", "-"):
            potong.append(f"Peran: {peran}")
        grup = getattr(r, "grup", None)
        if isinstance(grup, str) and grup not in ("", "-"):
            potong.append(f"Grup: {grup}")
        potong.append(f"Hop: {r.hop} · Klaster: {r.community_id}")
        if isinstance(r.pd, float) and not math.isnan(r.pd):
            potong.append(f"PD: {r.pd * 100:.2f}%")
        teks.append("<br>".join(potong))

    jejak.append(go.Scatter(
        x=[posisi[n][0] for n in nodes["id"]],
        y=[posisi[n][1] for n in nodes["id"]],
        mode="markers",
        marker=dict(size=ukuran, color=warna, symbol=simbol,
                    line=dict(width=1.4, color=garis_tepi)),
        text=teks,
        hovertemplate="%{text}<extra></extra>",
    ))

    fig = go.Figure(jejak)
    fig.update_layout(
        height=540, showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        hovermode="closest",
    )
    return fig


# --------------------------------------------------------------------------
# Visual model: meter PD, peta klaster, kurva recall
# --------------------------------------------------------------------------
def meter_pd(pd_nilai: float, cutoffs: dict, warna: str) -> go.Figure:
    """Meter PD dengan pita ambang risiko dari artefak model.

    Sumbu dipotong pada dua kali ambang q95 supaya pita risiko rendah tidak
    tergencet menjadi garis rambut pada portofolio dengan PD kecil.
    """
    maksimum = max(float(cutoffs["q95"]) * 2, pd_nilai * 1.25, 0.02)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pd_nilai * 100,
        number=dict(suffix="%", valueformat=".2f", font=dict(size=30, color=warna)),
        gauge=dict(
            axis=dict(range=[0, maksimum * 100], tickformat=".1f",
                      tickcolor="rgba(29,95,168,.35)"),
            bar=dict(color=warna, thickness=0.72),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0, cutoffs["q50"] * 100], color="rgba(31,138,95,.16)"),
                dict(range=[cutoffs["q50"] * 100, cutoffs["q80"] * 100],
                     color="rgba(224,160,42,.18)"),
                dict(range=[cutoffs["q80"] * 100, cutoffs["q95"] * 100],
                     color="rgba(226,104,63,.20)"),
                dict(range=[cutoffs["q95"] * 100, maksimum * 100],
                     color="rgba(204,59,82,.22)"),
            ],
            threshold=dict(line=dict(color=MERAH, width=3), thickness=0.8,
                           value=cutoffs["q95"] * 100),
        ),
    ))
    return gaya_plot(fig, 230)


def plot_klaster(ruang, posisi=None, contoh: int = 1400) -> go.Figure:
    """Peta klaster portofolio dengan posisi pengajuan baru di atasnya.

    Latar diambil sebagai cuplikan acak portofolio supaya bidangnya terbaca
    tanpa menggambar ribuan titik; pusat klaster dan pengajuan baru selalu
    digambar penuh.
    """
    titik = ruang.titik
    if len(titik) > contoh:
        titik = titik.sample(contoh, random_state=11)

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=titik.loc[titik["default"] == 0, "x"],
        y=titik.loc[titik["default"] == 0, "y"],
        mode="markers", name="Portofolio non-default",
        marker=dict(size=5, color="rgba(29,95,168,.30)", line=dict(width=0)),
        hovertemplate="Non-default<extra></extra>",
    ))
    fig.add_trace(go.Scattergl(
        x=titik.loc[titik["default"] == 1, "x"],
        y=titik.loc[titik["default"] == 1, "y"],
        mode="markers", name="Portofolio default",
        marker=dict(size=8, color="rgba(204,59,82,.75)", symbol="x",
                    line=dict(width=0)),
        hovertemplate="Default<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ruang.pusat["x"], y=ruang.pusat["y"], mode="markers+text",
        name="Pusat klaster",
        text=[f"{n}<br>{persen(t, 1)}" for n, t in
              zip(ruang.pusat["nama"], ruang.pusat["tingkat_default"])],
        textposition="top center",
        textfont=dict(size=10, color=PRIMER_TUA),
        marker=dict(size=17, color="#ffffff", symbol="circle",
                    line=dict(width=2.6, color=PRIMER_TUA)),
        hovertemplate="%{text}<extra></extra>",
    ))
    if posisi is not None:
        fig.add_trace(go.Scatter(
            x=[posisi.x], y=[posisi.y], mode="markers+text",
            name="Pengajuan ini", text=["Pengajuan ini"], textposition="bottom center",
            textfont=dict(size=12, color=AMBER),
            marker=dict(size=22, color=AMBER, symbol="star",
                        line=dict(width=1.6, color="#7a5406")),
            hovertemplate=f"Pengajuan ini<br>Klaster terdekat: {posisi.nama}<extra></extra>",
        ))
    fig.update_layout(
        xaxis_title=f"Komponen utama 1 ({ruang.varians[0]:.0%} ragam)",
        yaxis_title=f"Komponen utama 2 ({ruang.varians[1]:.0%} ragam)",
    )
    return gaya_plot(fig, 470)


def plot_jarak_klaster(jarak: pd.DataFrame) -> go.Figure:
    """Jarak pengajuan ke tiap pusat klaster — makin pendek makin mirip."""
    data = jarak.iloc[::-1]
    warna = [
        MERAH if t >= 0.05 else (AMBER if t >= 0.02 else AKSEN)
        for t in data["tingkat_default"]
    ]
    fig = go.Figure(go.Bar(
        x=data["jarak"], y=data["nama"], orientation="h", marker_color=warna,
        text=[f"default {persen(t, 1)}" for t in data["tingkat_default"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Jarak %{x:.2f}<br>%{text}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Jarak ke pusat klaster (ruang fitur baku)",
                      yaxis_title=None, showlegend=False)
    return gaya_plot(fig, 240)


def plot_recall_ambang(kurva: pd.DataFrame, judul: str = "") -> go.Figure:
    """Recall, presisi, dan porsi alarm pada tiap ambang operasional."""
    fig = go.Figure()
    seri = [("recall", "Recall", AKSEN), ("presisi", "Presisi", PRIMER),
            ("porsi_alarm", "Porsi berkas dialarmkan", AMBER)]
    for kolom, nama, warna in seri:
        fig.add_trace(go.Bar(
            x=kurva["ambang"], y=kurva[kolom], name=nama, marker_color=warna,
            text=[f"{v:.0%}" for v in kurva[kolom]], textposition="outside",
            hovertemplate=f"<b>{nama}</b><br>%{{x}}: %{{y:.1%}}<extra></extra>",
        ))
    fig.update_layout(barmode="group", yaxis_tickformat=".0%",
                      yaxis_title=None, xaxis_title="Ambang operasional",
                      title=judul, yaxis_range=[0, 1.05])
    return gaya_plot(fig, 360)
