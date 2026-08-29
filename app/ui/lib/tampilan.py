"""Komponen tampilan yang dipakai bersama oleh seluruh halaman.

Satu berkas ini memegang seluruh keputusan rupa aplikasi: palet, lembar gaya,
kepala halaman, kartu, dan bentuk baku setiap grafik. Halaman tidak boleh
menuliskan kode warna sendiri — mereka mengimpor nama dari sini, supaya
mengganti satu warna cukup dilakukan di satu tempat.

Palet resmi: jingga #FF8000, tosca #40C0C0, abu #808080, putih #FFFFFF.

Ikon dan emoji sengaja hampir tidak dipakai. Yang membedakan satu bagian dari
bagian lain adalah warna pita, tebal huruf, dan jarak — bukan gambar kecil di
depan judul.
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
# Empat warna induk: jingga, tosca, abu, putih. Sisanya hanya gelap-terang dari
# keempatnya, bukan warna baru.
#
# Pembagian tugasnya tetap: tosca membawa struktur (kepala halaman, batang
# grafik, sisi baik dari sebuah ukuran), jingga membawa tindakan dan peringatan,
# abu membawa segala yang netral, putih membawa permukaan. Tingkat risiko dibaca
# sebagai satu tanjakan tosca -> jingga -> jingga gelap, sehingga urutannya
# terbaca meski dicetak hitam putih atau dilihat mata yang sulit membedakan
# merah dan hijau.
JINGGA = "#FF8000"
JINGGA_MUDA = "#FFA94D"
JINGGA_TUA = "#B35900"
JINGGA_GELAP = "#7A3C00"

TOSCA = "#40C0C0"
TOSCA_MUDA = "#8FD9D9"
TOSCA_TUA = "#2A8080"
TOSCA_GELAP = "#1A5252"

ABU = "#808080"
ABU_MUDA = "#D9DBDC"
ABU_LATAR = "#F5F6F7"
ABU_TUA = "#4D4D4D"
TINTA = "#2E3233"
PUTIH = "#FFFFFF"

# Tanjakan risiko: baik -> perhatian -> buruk.
DERET_RISIKO = [TOSCA_TUA, TOSCA, JINGGA_MUDA, JINGGA, JINGGA_GELAP]
# Deret kategori: tosca dan jingga berselang-seling supaya dua kategori yang
# bersebelahan tidak pernah sewarna, ditutup abu untuk sisa kategori.
DERET_KATEGORI = [TOSCA, JINGGA, TOSCA_GELAP, JINGGA_TUA, ABU,
                  TOSCA_MUDA, JINGGA_GELAP, ABU_TUA]

GAYA = """
<style>
  :root {
    --jingga:#FF8000; --jingga-tua:#B35900; --jingga-gelap:#7A3C00;
    --tosca:#40C0C0; --tosca-tua:#2A8080; --tosca-gelap:#1A5252;
    --abu:#808080; --abu-muda:#D9DBDC; --abu-latar:#F5F6F7; --tinta:#2E3233;
    --garis:rgba(128,128,128,.28); --lembut:rgba(64,192,192,.10);
  }
  .block-container {padding-top:1.5rem; padding-bottom:3.5rem; max-width:1400px;}
  h1, h2, h3 {letter-spacing:-.015em;}

  /* Kepala halaman. Satu bidang tosca gelap dengan garis jingga di kiri —
     tanpa gradasi ramai, supaya terbaca sebagai kop dokumen, bukan spanduk. */
  .hero {display:flex; gap:1.15rem; align-items:flex-start;
         background:linear-gradient(120deg,var(--tosca-gelap) 0%,var(--tosca-tua) 100%);
         border-left:5px solid var(--jingga);
         color:#fff; border-radius:10px; padding:1.15rem 1.4rem; margin-bottom:1.15rem;}
  .hero-nomor {font-size:1.5rem; font-weight:700; line-height:1.1; letter-spacing:.02em;
               color:#fff; background:rgba(255,255,255,.12);
               border:1px solid rgba(255,255,255,.22); border-radius:8px;
               padding:.5rem .75rem; min-width:3rem; text-align:center;
               white-space:nowrap;}
  .hero-teks h1 {font-size:1.42rem; margin:0 0 .25rem 0; color:#fff; font-weight:700;
                 padding:0;}
  .hero-teks p {margin:0; font-size:.9rem; color:rgba(255,255,255,.88); line-height:1.5;
                max-width:76ch;}
  .hero-chips {margin-top:.7rem; display:flex; flex-wrap:wrap; gap:.4rem;}
  .hero-chip {background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.24);
              border-radius:4px; padding:.18rem .6rem; font-size:.74rem;
              color:rgba(255,255,255,.9);}
  .hero-chip b {font-weight:700; color:#fff;}

  /* Judul bagian */
  .bagian {margin:1.6rem 0 .6rem 0; border-left:3px solid var(--jingga); padding-left:.7rem;}
  .bagian h3 {margin:0; font-size:1.02rem; font-weight:700; color:var(--tinta);}
  .bagian p {margin:.18rem 0 0 0; font-size:.82rem; color:var(--abu);}

  /* Metrik */
  div[data-testid="stMetric"] {background:#fff; border:1px solid var(--garis);
        border-top:3px solid var(--tosca); border-radius:8px; padding:.75rem .9rem;}
  div[data-testid="stMetricValue"] {font-size:1.38rem; font-weight:700; color:var(--tinta);}
  div[data-testid="stMetricLabel"] p {font-size:.78rem; color:#5c6366; font-weight:600;}
  div[data-testid="stMetricLabel"] {overflow:visible; white-space:normal;}

  /* Kartu dan lencana */
  .badge {display:inline-block; padding:.2rem .6rem; border-radius:4px;
          font-size:.73rem; font-weight:700; letter-spacing:.03em;}
  .kotak {background:#fff; border:1px solid var(--garis); border-radius:8px;
          padding:.85rem 1rem; margin-bottom:.6rem;}
  .kartu {background:#fff; border:1px solid var(--garis); border-left:4px solid var(--w);
          border-radius:8px; padding:.8rem .95rem; margin-bottom:.55rem;}
  .kartu-judul {font-weight:700; font-size:.9rem; color:var(--tinta);}
  .kartu-isi {font-size:.85rem; color:#5c6366; margin-top:.2rem; line-height:1.55;}
  .gerbang {background:#fff; border:1px solid var(--garis); border-left:4px solid var(--w);
            border-radius:8px; padding:.8rem .95rem; margin-bottom:.55rem;}
  .gerbang .aspek {font-weight:700; letter-spacing:.06em; text-transform:uppercase;
            font-size:.69rem;}
  .gerbang .pasal {font-size:.74rem; color:var(--abu); font-family:ui-monospace,monospace;}
  .tipis {color:var(--abu); font-size:.84rem;}

  /* Titik status pada sidebar — pengganti emoji lampu */
  .titik {display:inline-block; width:.55rem; height:.55rem; border-radius:50%;
          margin-right:.45rem; vertical-align:middle;}
  .titik-siap {background:var(--tosca);}
  .titik-kurang {background:var(--abu-muda); border:1px solid var(--abu);}
  .baris-status {font-size:.85rem; margin:.18rem 0; color:var(--tinta);}

  /* Tab */
  button[data-baseweb="tab"] {font-weight:600;}
  div[data-baseweb="tab-list"] {gap:.25rem; border-bottom:1px solid var(--garis);}
  button[data-baseweb="tab"][aria-selected="true"] {background:var(--lembut);
        border-radius:6px 6px 0 0;}
  div[data-baseweb="tab-highlight"] {background:var(--jingga);}

  /* Sidebar */
  section[data-testid="stSidebar"] {background:var(--abu-latar);
        border-right:1px solid var(--garis);}
  .sidebar-merek {font-size:.98rem; line-height:1.25; color:var(--tosca-gelap);
        border-bottom:2px solid var(--jingga); padding-bottom:.55rem; margin-bottom:.7rem;
        text-transform:uppercase; letter-spacing:.06em;}
  .sidebar-merek b {font-size:1.02rem; letter-spacing:.02em;}

  /* Tombol */
  button[kind="primary"] {border-radius:6px; font-weight:700;
        background:var(--jingga); border-color:var(--jingga);}
  button[kind="primary"]:hover {background:var(--jingga-tua); border-color:var(--jingga-tua);}
  button[kind="secondary"] {border-radius:6px;}

  /* Tabel dan expander */
  div[data-testid="stDataFrame"] {border-radius:8px; overflow:hidden;
        border:1px solid var(--garis);}
  details[data-testid="stExpander"] {border-radius:8px; border:1px solid var(--garis);}
</style>
"""


def gaya_plot(fig, tinggi: int | None = None):
    """Satu selera untuk semua grafik: latar bersih, kisi tipis, huruf sama."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Source Sans Pro, sans-serif", size=12, color=TINTA),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(128,128,128,.04)",
        margin=dict(l=10, r=10, t=44, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=PUTIH, bordercolor="rgba(128,128,128,.45)"),
    )
    # Gaya judul hanya dipasang bila figure memang punya judul; menyetel
    # title_font pada figure tanpa judul membuat Plotly mencetak "undefined".
    if fig.layout.title.text:
        fig.update_layout(title=dict(font=dict(size=14, color=TINTA), x=0, xanchor="left"))
    fig.update_xaxes(gridcolor="rgba(128,128,128,.16)", zerolinecolor="rgba(128,128,128,.35)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,.16)", zerolinecolor="rgba(128,128,128,.35)")
    if tinggi:
        fig.update_layout(height=tinggi)
    return fig


PALET_TIPE = {
    "Badan hukum": TOSCA_TUA,
    "Grup usaha": TOSCA_GELAP,
    "Pemilik manfaat": JINGGA,
    "Pengurus": JINGGA_TUA,
    "Counterparty": TOSCA,
    "Atribut berbagi": JINGGA_GELAP,
    "Agunan": ABU,
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

# Klaster graf: tosca dan jingga bergantian pada beberapa tingkat kecerahan,
# ditutup abu. Dua belas cukup untuk subgraf sebesar apa pun yang ditampilkan.
PALET_KOMUNITAS = [
    TOSCA, JINGGA, TOSCA_GELAP, JINGGA_TUA, ABU, TOSCA_MUDA,
    JINGGA_GELAP, ABU_TUA, TOSCA_TUA, JINGGA_MUDA, "#6FA8A8", "#A67142",
]

WARNA_STATUS = {
    mock_engine.LOLOS: TOSCA_TUA,
    mock_engine.TELAAH: JINGGA,
    mock_engine.PENYESUAIAN: JINGGA_GELAP,
}

WARNA_KEPUTUSAN = {
    "SETUJU": TOSCA_TUA,
    "SETUJU DENGAN SYARAT": TOSCA,
    "PERLU PENYESUAIAN": JINGGA,
    "TOLAK": JINGGA_GELAP,
}


def setup_halaman(judul: str, ikon: str = "◆") -> None:
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


def kartu(judul: str, isi: str, warna: str = TOSCA) -> str:
    """Kartu berpita warna untuk temuan, pola, dan catatan pendek."""
    return (
        f'<div class="kartu" style="--w:{warna}">'
        f'<div class="kartu-judul">{judul}</div>'
        f'<div class="kartu-isi">{isi}</div></div>'
    )


def baris_status(label: str, siap: bool, catatan: str = "") -> str:
    """Satu baris kesiapan lapisan: titik berwarna, bukan emoji lampu."""
    kelas = "titik-siap" if siap else "titik-kurang"
    ekor = "" if siap or not catatan else f" <span class='tipis'>· {catatan}</span>"
    return f'<div class="baris-status"><span class="titik {kelas}"></span>{label}{ekor}</div>'


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
        ("Eksposur grup berjalan", eksposur_grup, TOSCA_TUA),
        ("Usulan fasilitas ini", tambahan, JINGGA),
        ("Sisa ruang BMPK", sisa, "rgba(128,128,128,.25)"),
    ]
    for nama, nilai, warna in potongan:
        fig.add_trace(go.Bar(
            x=[nilai / 1e9], y=["BMPK grup"], name=nama, orientation="h",
            marker_color=warna,
            hovertemplate=f"<b>{nama}</b><br>Rp %{{x:.1f}} M<extra></extra>",
        ))
    fig.add_vline(x=batas / 1e9, line_dash="dash", line_color=JINGGA_GELAP,
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
            marker_color=[JINGGA if k.dampak > 0 else TOSCA_TUA for k in data],
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
            color=[TOSCA_TUA] + [
                JINGGA if "Pemilik manfaat" in j else TOSCA
                for j in rantai["jenis"]
            ],
        ),
        link=dict(
            source=[indeks[r["pemilik"]] for _, r in rantai.iterrows()],
            target=[indeks[r["dimiliki"]] for _, r in rantai.iterrows()],
            value=[float(r["porsi_langsung"]) * 100 for _, r in rantai.iterrows()],
            color="rgba(64,192,192,.35)",
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
                color="rgba(42,128,128,.60)" if struktural else "rgba(128,128,128,.35)",
            ),
        ))

    if warnai == "komunitas":
        warna = [PALET_KOMUNITAS[int(c) % len(PALET_KOMUNITAS)] for c in nodes["community_id"]]
    else:
        warna = [PALET_TIPE.get(t, "#888888") for t in nodes["tipe"]]

    ukuran = [30 if nid == sorot else (18 if h <= 1 else 12)
              for nid, h in zip(nodes["id"], nodes["hop"])]
    garis_tepi = ["#2E3233" if nid == sorot else "rgba(255,255,255,.75)" for nid in nodes["id"]]
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
                      tickcolor="rgba(128,128,128,.45)"),
            bar=dict(color=warna, thickness=0.72),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0, cutoffs["q50"] * 100], color="rgba(64,192,192,.20)"),
                dict(range=[cutoffs["q50"] * 100, cutoffs["q80"] * 100],
                     color="rgba(255,169,77,.20)"),
                dict(range=[cutoffs["q80"] * 100, cutoffs["q95"] * 100],
                     color="rgba(255,128,0,.24)"),
                dict(range=[cutoffs["q95"] * 100, maksimum * 100],
                     color="rgba(122,60,0,.26)"),
            ],
            threshold=dict(line=dict(color=JINGGA_GELAP, width=3), thickness=0.8,
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
        marker=dict(size=5, color="rgba(42,128,128,.35)", line=dict(width=0)),
        hovertemplate="Non-default<extra></extra>",
    ))
    fig.add_trace(go.Scattergl(
        x=titik.loc[titik["default"] == 1, "x"],
        y=titik.loc[titik["default"] == 1, "y"],
        mode="markers", name="Portofolio default",
        marker=dict(size=8, color="rgba(255,128,0,.85)", symbol="x",
                    line=dict(width=0)),
        hovertemplate="Default<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ruang.pusat["x"], y=ruang.pusat["y"], mode="markers+text",
        name="Pusat klaster",
        text=[f"{n}<br>{persen(t, 1)}" for n, t in
              zip(ruang.pusat["nama"], ruang.pusat["tingkat_default"])],
        textposition="top center",
        textfont=dict(size=10, color=TOSCA_GELAP),
        marker=dict(size=17, color="#ffffff", symbol="circle",
                    line=dict(width=2.6, color=TOSCA_GELAP)),
        hovertemplate="%{text}<extra></extra>",
    ))
    if posisi is not None:
        fig.add_trace(go.Scatter(
            x=[posisi.x], y=[posisi.y], mode="markers+text",
            name="Pengajuan ini", text=["Pengajuan ini"], textposition="bottom center",
            textfont=dict(size=12, color=JINGGA_TUA),
            marker=dict(size=22, color=JINGGA, symbol="star",
                        line=dict(width=1.6, color=JINGGA_GELAP)),
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
        JINGGA_GELAP if t >= 0.05 else (JINGGA if t >= 0.02 else TOSCA_TUA)
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
    seri = [("recall", "Recall", TOSCA_TUA), ("presisi", "Presisi", TOSCA),
            ("porsi_alarm", "Porsi berkas dialarmkan", JINGGA)]
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
