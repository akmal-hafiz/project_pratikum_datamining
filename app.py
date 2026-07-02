import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindScope · Student Depression AI",
    page_icon=":material/track_changes:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Icon set (hand-drawn, stroke-based, monochrome) ─────────────────────────
_ICON_PATHS = {
    'mark':      '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4.2"/><circle cx="12" cy="12" r=".7" fill="currentColor" stroke="none"/>',
    'home':      '<path d="M3 11.2 12 4l9 7.2"/><path d="M5.2 9.8V20h13.6V9.8"/>',
    'upload':    '<path d="M12 4v11"/><path d="M7.2 8.8 12 4l4.8 4.8"/><path d="M4 20h16"/>',
    'chart':     '<path d="M4 20V11"/><path d="M10.7 20V4"/><path d="M17.3 20v-7.5"/><path d="M3 20h18"/>',
    'cpu':       '<rect x="7" y="7" width="10" height="10" rx=".5"/><path d="M7 3.2v3.3M12 3.2v3.3M17 3.2v3.3M7 17.5v3.3M12 17.5v3.3M17 17.5v3.3M3.2 7h3.3M3.2 12h3.3M3.2 17h3.3M17.5 7h3.3M17.5 12h3.3M17.5 17h3.3"/>',
    'target':    '<circle cx="12" cy="12" r="8.2"/><circle cx="12" cy="12" r="4.4"/><circle cx="12" cy="12" r=".7" fill="currentColor" stroke="none"/>',
    'check':     '<circle cx="12" cy="12" r="9"/><path d="M7.8 12.3 10.6 15l5.6-6.4"/>',
    'alert':     '<path d="M12 3 22 20H2Z"/><path d="M12 9.2v5"/><circle cx="12" cy="17" r=".7" fill="currentColor" stroke="none"/>',
    'info':      '<circle cx="12" cy="12" r="9"/><path d="M12 11v5.5"/><circle cx="12" cy="7.4" r=".7" fill="currentColor" stroke="none"/>',
    'arrow':     '<path d="M4 12h16"/><path d="M13.5 5.5 20 12l-6.5 6.5"/>',
    'folder':    '<path d="M3.2 6.4h6l1.8 2h9.8v10.2H3.2Z"/>',
    'compass':   '<circle cx="12" cy="12" r="9"/><path d="M15.5 8.5 13.4 13.4 8.5 15.5 10.6 10.6Z"/>',
    'lock':      '<rect x="5.5" y="10.5" width="13" height="9" rx=".5"/><path d="M8.2 10.5V7.8a3.8 3.8 0 0 1 7.6 0v2.7"/>',
    'x':         '<path d="M6 6l12 12"/><path d="M18 6 6 18"/>',
}

def ic(name: str, size: int = 18, stroke: float = 1.6) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="{stroke}" stroke-linecap="round" '
            f'stroke-linejoin="round" style="display:inline-block;vertical-align:-3px">'
            f'{_ICON_PATHS[name]}</svg>')

def sub_header(text: str, help_text: str | None = None):
    """Bold sub-header with an optional click-to-read info popover."""
    if not help_text:
        st.markdown(f"**{text}**")
        return
    c1, c2 = st.columns([0.93, 0.07])
    with c1:
        st.markdown(f"**{text}**")
    with c2:
        with st.popover("", icon=":material/help:", use_container_width=True):
            st.markdown(f"<div style='font-size:.82rem;color:var(--text-dim);line-height:1.65'>{help_text}</div>",
                        unsafe_allow_html=True)


# ── Global CSS — monochrome, hairline, zero gradient ────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --ink:        #000000;
    --paper:      #ffffff;
    --surface:    #0c0c0c;
    --surface-2:  #131313;
    --line:       #2b2b2b;
    --line-soft:  #1c1c1c;
    --text:       #f2f2f2;
    --text-dim:   #9a9a9a;
    --text-faint: #5c5c5c;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.stApp { background-color: var(--ink); color: var(--text); }

[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] > div { padding-top: 0; }

#MainMenu, footer, header { visibility: hidden; }

h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif; }

/* ── Motion (respects reduced-motion) ── */
@keyframes riseIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }
.rise { animation: riseIn .5s ease both; }
@media (prefers-reduced-motion: reduce) { .rise { animation: none; } }

/* ── KPI cards ── */
.kpi-card {
    background: var(--surface-2);
    border: 1px solid var(--line);
    border-radius: 0;
    padding: 22px 20px;
    text-align: left;
    position: relative;
}
.kpi-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: var(--paper);
    line-height: 1;
}
.kpi-label {
    font-size: .7rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: .12em;
    margin-top: 8px;
    font-weight: 500;
}

/* ── Section headers ── */
.sec-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--paper);
    border-left: 2px solid var(--paper);
    padding-left: 12px;
    margin: 30px 0 16px;
    letter-spacing: .01em;
}
.eyebrow {
    font-size: .68rem;
    font-weight: 600;
    color: var(--text-dim);
    letter-spacing: .18em;
    text-transform: uppercase;
}

/* ── Info / status boxes (monochrome, weight-differentiated) ── */
.box-info {
    background: var(--surface-2);
    border: 1px solid var(--line);
    border-radius: 0;
    padding: 16px 18px;
    color: var(--text-dim);
    font-size: .87rem;
    line-height: 1.6;
}
.box-danger {
    background: var(--paper);
    color: var(--ink);
    border: 1px solid var(--paper);
    border-radius: 0;
    padding: 28px 24px;
    text-align: center;
}
.box-success {
    background: transparent;
    border: 1px solid var(--paper);
    border-radius: 0;
    padding: 28px 24px;
    text-align: center;
    color: var(--paper);
}

/* ── Probability bars ── */
.prob-wrap { margin: 8px 0 14px; }
.prob-track {
    background: var(--line-soft);
    border-radius: 0;
    height: 8px;
    overflow: hidden;
    margin-top: 5px;
    border: 1px solid var(--line);
}
.prob-fill { height: 100%; }

/* ── Step / usage-flow cards ── */
.step-card {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    background: var(--surface-2);
    border: 1px solid var(--line);
    border-radius: 0;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.step-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .72rem;
    color: var(--text-faint);
    border: 1px solid var(--line);
    width: 26px; height: 26px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.step-icon {
    color: var(--paper);
    flex-shrink: 0;
    margin-top: 1px;
}

/* ── Buttons: solid inversion, zero radius, zero gradient ── */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
    background: var(--paper) !important;
    color: var(--ink) !important;
    border: 1px solid var(--paper) !important;
    border-radius: 0 !important;
    padding: 10px 22px !important;
    font-weight: 600 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    transition: background .15s, color .15s !important;
    width: 100% !important;
}
.stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {
    background: var(--ink) !important;
    color: var(--paper) !important;
    border-color: var(--paper) !important;
}

/* Sidebar nav buttons: outline default, filled = current page (via disabled) */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: var(--text-dim) !important;
    border: 1px solid transparent !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 10px 14px !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--surface-2) !important;
    color: var(--paper) !important;
    border-color: var(--line) !important;
}
[data-testid="stSidebar"] .stButton > button:disabled {
    background: var(--paper) !important;
    color: var(--ink) !important;
    border-color: var(--paper) !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}

/* ── Form inputs ── */
.stSelectbox label, .stSlider label, .stNumberInput label, .stTextInput label {
    color: var(--text-dim) !important;
    font-size: .8rem !important;
    font-weight: 500 !important;
}
.stSelectbox > div > div, .stNumberInput input, .stTextInput input {
    background-color: var(--surface-2) !important;
    border-color: var(--line) !important;
    border-radius: 0 !important;
    color: var(--text) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background-color: var(--surface-2) !important;
    border: 1px dashed var(--line) !important;
    border-radius: 0 !important;
}

/* Dataframes / tables: sharp corners */
[data-testid="stDataFrame"] { border: 1px solid var(--line); }

hr { border-color: var(--line) !important; }

/* ── Info popovers (round, quiet, distinct from primary buttons) ── */
[data-testid="stPopover"] button {
    width: 30px !important;
    height: 30px !important;
    min-width: 30px !important;
    padding: 0 !important;
    margin-top: 2px !important;
    border-radius: 50% !important;
    background: transparent !important;
    border: 1px solid var(--line) !important;
    color: var(--text-dim) !important;
}
[data-testid="stPopover"] button:hover {
    border-color: var(--paper) !important;
    color: var(--paper) !important;
    background: var(--surface-2) !important;
}
[data-testid="stPopoverBody"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--line) !important;
    border-radius: 0 !important;
}

/* ── Tutorial dialog ── */
[data-testid="stDialog"] [data-testid="stVerticalBlockBorderWrapper"] { background: var(--surface); }
</style>
""", unsafe_allow_html=True)


# ── Session state ────────────────────────────────────────────────────────────
def _ss():
    defaults = dict(
        page='dashboard',
        df=None, df_proc=None, le={},
        model=None, model_name=None, feature_cols=None,
        train_acc=None, test_acc=None,
        report=None, cm=None, feat_imp=None,
        pred=None, proba=None,
        show_tutorial=True, tut_step=0,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_ss()


# ── First-run tutorial (step-by-step onboarding) ────────────────────────────
TUT_STEPS = [
    ('mark', 'Selamat datang di MindScope',
     'Aplikasi ini memprediksi risiko depresi mahasiswa dari data akademik dan gaya hidup, '
     'memakai machine learning. Dibuat untuk tugas data mining — bukan alat diagnosis klinis.'),
    ('upload', 'Mulai dari data',
     'Unggah dataset CSV di halaman Data &amp; Eksplorasi. Sistem otomatis membersihkan nilai '
     'kosong dan mengubah kategori jadi angka yang bisa dibaca model.'),
    ('chart', 'Baca polanya',
     'Halaman Visualisasi menunjukkan hubungan tekanan akademik, durasi tidur, dan faktor lain '
     'terhadap risiko depresi lewat grafik interaktif.'),
    ('cpu', 'Latih modelnya',
     'Pilih Decision Tree atau Naive Bayes di halaman Training Model, atur parameter, lalu '
     'latih. Akurasi dan confusion matrix muncul otomatis.'),
    ('target', 'Coba prediksi',
     'Isi profil mahasiswa di halaman Prediksi untuk melihat estimasi risiko beserta '
     'persentase kepastiannya.'),
]

@st.dialog("Tur Aplikasi", width="large")
def _show_tutorial():
    step = st.session_state.tut_step
    name, title, desc = TUT_STEPS[step]
    st.markdown(f"""
    <div style='text-align:center;padding:6px 8px 4px'>
        <div style='color:var(--paper)'>{ic(name, 40, 1.3)}</div>
        <div style="font-family:'Fraunces',serif;font-size:1.5rem;font-weight:500;
            color:var(--paper);margin-top:16px">{title}</div>
        <div style='color:var(--text-dim);font-size:.87rem;line-height:1.7;margin-top:12px;
            max-width:420px;margin-left:auto;margin-right:auto'>{desc}</div>
    </div>
    """, unsafe_allow_html=True)

    dots = "".join(
        f"<span style='display:inline-block;width:6px;height:6px;border-radius:50%;margin:0 3px;"
        f"background:{'var(--paper)' if i == step else 'var(--line)'}'></span>"
        for i in range(len(TUT_STEPS))
    )
    st.markdown(f"<div style='text-align:center;margin:22px 0 6px'>{dots}</div>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns([1, 1, 1])
    with b1:
        if step > 0 and st.button("Kembali", key="tut_back", use_container_width=True):
            st.session_state.tut_step -= 1
            st.rerun()
    with b2:
        if st.button("Lewati", key="tut_skip", use_container_width=True):
            st.session_state.show_tutorial = False
            st.rerun()
    with b3:
        label = "Mulai" if step == len(TUT_STEPS) - 1 else "Lanjut"
        if st.button(label, key="tut_next", use_container_width=True):
            if step == len(TUT_STEPS) - 1:
                st.session_state.show_tutorial = False
            else:
                st.session_state.tut_step += 1
            st.rerun()

if st.session_state.show_tutorial:
    _show_tutorial()


# ── Preprocessing ────────────────────────────────────────────────────────────
SLEEP_MAP = {
    'Less than 5 hours': 1,
    '5-6 hours': 2,
    '7-8 hours': 3,
    'More than 8 hours': 4,
}
DIET_MAP  = {'Unhealthy': 0, 'Moderate': 1, 'Healthy': 2}
BIN_MAP   = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}

def preprocess(df: pd.DataFrame):
    df = df.copy()
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    # Profession is near-constant (99.9% Student) — drop to reduce noise
    if 'Profession' in df.columns:
        df = df.drop('Profession', axis=1)

    for c in df.select_dtypes('number').columns:
        df[c] = df[c].fillna(df[c].median())
    for c in df.select_dtypes('object').columns:
        df[c] = df[c].fillna(df[c].mode()[0])

    le_store = {}

    for c in ['Gender', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']:
        if c in df.columns:
            df[c] = df[c].map(BIN_MAP).fillna(0).astype(int)

    if 'Sleep Duration' in df.columns:
        df['Sleep Duration'] = df['Sleep Duration'].map(SLEEP_MAP).fillna(2).astype(int)
    if 'Dietary Habits'  in df.columns:
        df['Dietary Habits'] = df['Dietary Habits'].map(DIET_MAP).fillna(1).astype(int)

    for c in ['City', 'Degree']:
        if c in df.columns:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            le_store[c] = le

    return df, le_store


# ── Monochrome plot theme ────────────────────────────────────────────────────
GREY = '#8c8c8c'
LINE = '#242424'
PT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(19,19,19,.9)',
    font_color='#b3b3b3',
    title_font_color='#f2f2f2',
    font_family='IBM Plex Sans',
    margin=dict(t=24, b=24, l=24, r=24),
)
AXIS = dict(
    xaxis=dict(gridcolor=LINE, linecolor='#333333', tickfont_color='#7a7a7a'),
    yaxis=dict(gridcolor=LINE, linecolor='#333333', tickfont_color='#7a7a7a'),
)
# class distinction: color where it carries meaning, UI chrome stays monochrome
MONO = {'Tidak Depresi': '#5c7cfa', 'Depresi': '#ff6b6b'}
MONO_SEQ = ['#5c7cfa', '#ff6b6b']
PATTERN = {'Tidak Depresi': '', 'Depresi': '/'}
FEAT_SCALE = ['#0b3d3d', '#20c997']


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — custom icon nav (Material Symbols via native `icon=`)
# ═══════════════════════════════════════════════════════════════════════════
NAV = [
    ('dashboard', 'Dashboard',           'dashboard'),
    ('data',      'Data & Eksplorasi',   'upload_file'),
    ('viz',       'Visualisasi',         'bar_chart'),
    ('train',     'Training Model',      'model_training'),
    ('predict',   'Prediksi',            'track_changes'),
]

with st.sidebar:
    st.markdown(f"""
    <div style='padding:28px 20px 18px;display:flex;align-items:center;gap:10px'>
        <span style="color:var(--paper)">{ic('mark', 26, 1.4)}</span>
        <div>
            <div style='font-size:1rem;font-weight:600;color:var(--paper);letter-spacing:.01em'>MindScope</div>
            <div style='font-size:.64rem;color:var(--text-faint);letter-spacing:.1em;text-transform:uppercase'>Depression Risk Model</div>
        </div>
    </div>
    <hr style='margin:0 0 12px'>
    """, unsafe_allow_html=True)

    for key, label, micon in NAV:
        active = st.session_state.page == key
        if st.button(label, key=f'nav_{key}', icon=f':material/{micon}:',
                     disabled=active, use_container_width=True):
            st.session_state.page = key

    st.markdown("<hr style='margin-top:24px'>", unsafe_allow_html=True)
    if st.button("Tampilkan Tutorial", key="nav_replay", icon=":material/help:", use_container_width=True):
        st.session_state.tut_step = 0
        st.session_state.show_tutorial = True
        st.rerun()

    st.markdown(f"""<div style='font-size:.66rem;color:var(--text-faint);padding:10px 4px;line-height:1.7'>
        DATA MINING PROJECT<br>Kaggle · Student Depression Dataset
    </div>""", unsafe_allow_html=True)

page = st.session_state.page


# ═══════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
if page == 'dashboard':
    st.markdown(f"""
    <div class='rise' style='padding:56px 0 36px'>
        <div class='eyebrow'>Data Mining Project</div>
        <h1 style="font-family:'Fraunces',serif;font-size:3rem;font-weight:500;margin:16px 0 0;
            line-height:1.12;color:var(--paper);max-width:720px">
            Student Depression Risk Predictor
        </h1>
        <p style='color:var(--text-dim);margin-top:18px;font-size:.95rem;max-width:560px;line-height:1.75'>
            Memprediksi risiko depresi mahasiswa dari data psikologis, akademik, dan gaya hidup —
            menggunakan Decision Tree dan Naive Bayes secara berdampingan.
        </p>
    </div>
    """, unsafe_allow_html=True)

    df = st.session_state.df
    c1, c2, c3, c4 = st.columns(4)
    vals = [
        (f"{len(df):,}" if df is not None else "—", "Total Data"),
        ((len(df.columns) - 1) if df is not None else "—", "Jumlah Fitur"),
        (f"{st.session_state.train_acc*100:.1f}%" if st.session_state.train_acc else "—", "Akurasi Train"),
        (f"{st.session_state.test_acc*100:.1f}%"  if st.session_state.test_acc  else "—", "Akurasi Test"),
    ]
    for col, (v, lbl) in zip([c1, c2, c3, c4], vals):
        with col:
            st.markdown(f"""<div class='kpi-card'>
                <div class='kpi-val'>{v}</div>
                <div class='kpi-label'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    la, lb = st.columns(2)

    with la:
        st.markdown("<div class='sec-header'>Tentang Aplikasi</div>", unsafe_allow_html=True)
        st.markdown("""<div class='box-info'>
            Aplikasi ini menganalisis faktor-faktor yang mempengaruhi risiko depresi mahasiswa —
            tekanan akademik, CGPA, durasi tidur, stres finansial, dan lainnya.<br><br>
            Dua algoritma tersedia: <b style='color:var(--paper)'>Decision Tree</b> (dapat diinterpretasi,
            feature importance) dan <b style='color:var(--paper)'>Naive Bayes</b> (probabilistik, cepat).
            Bandingkan keduanya di halaman Training Model.
        </div>""", unsafe_allow_html=True)

    with lb:
        st.markdown("<div class='sec-header'>Alur Penggunaan</div>", unsafe_allow_html=True)
        for n, (title, desc) in enumerate([
            ("Upload Dataset", "Unggah CSV di halaman Data & Eksplorasi"),
            ("Eksplorasi & Visualisasi", "Lihat distribusi dan korelasi fitur"),
            ("Training Model", "Pilih algoritma, atur parameter, latih"),
            ("Prediksi", "Masukkan profil mahasiswa, hasil instan"),
        ], start=1):
            st.markdown(f"""<div class='step-card'>
                <span class='step-num'>{n:02d}</span>
                <div>
                    <div style='font-weight:600;color:var(--paper);font-size:.87rem'>{title}</div>
                    <div style='color:var(--text-faint);font-size:.78rem;margin-top:2px'>{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    if df is None:
        st.markdown(f"""<br><div style='padding:18px 20px;border:1px solid var(--line);
            display:flex;align-items:center;gap:12px;color:var(--text-dim);font-size:.87rem'>
            {ic('arrow', 16)} Mulai dengan upload dataset di halaman <b style='color:var(--paper)'>&nbsp;Data & Eksplorasi</b>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# DATA & EKSPLORASI
# ═══════════════════════════════════════════════════════════════════════════
elif page == 'data':
    st.markdown("<div class='sec-header'>Upload & Eksplorasi Dataset</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload file CSV", type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state.df = df
        df_proc, le = preprocess(df)
        st.session_state.df_proc = df_proc
        st.session_state.le = le
        st.session_state.model = None
        st.success(f"Dataset dimuat: **{len(df):,} baris** · **{len(df.columns)} kolom**",
                   icon=":material/check_circle:")

    if st.session_state.df is not None:
        df = st.session_state.df

        st.markdown("<div class='sec-header'>Preview (5 Baris Pertama)</div>", unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        missing_total = int(df.isnull().sum().sum())
        depresi_pct   = f"{df['Depression'].mean()*100:.1f}%" if 'Depression' in df.columns else "—"
        for col, (v, lbl) in zip([c1,c2,c3,c4], [
            (f"{len(df):,}", "Total Baris"),
            (len(df.columns), "Total Kolom"),
            (missing_total, "Missing Values"),
            (depresi_pct, "Persen Depresi"),
        ]):
            with col:
                st.markdown(f"""<div class='kpi-card'>
                    <div class='kpi-val'>{v}</div>
                    <div class='kpi-label'>{lbl}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        la, lb = st.columns(2)

        with la:
            st.markdown("<div class='sec-header'>Distribusi Target (Depression)</div>", unsafe_allow_html=True)
            if 'Depression' in df.columns:
                cnt = df['Depression'].value_counts().reset_index()
                cnt.columns = ['Label', 'Jumlah']
                cnt['Label'] = cnt['Label'].map({0:'Tidak Depresi', 1:'Depresi'})
                fig = px.pie(cnt, values='Jumlah', names='Label',
                             color='Label', color_discrete_map=MONO, hole=.55)
                fig.update_traces(marker=dict(line=dict(color='#000000', width=2)), textfont_color='#0a0a0a')
                fig.update_layout(**PT, legend=dict(bgcolor='rgba(0,0,0,0)', font_color='#b3b3b3'))
                st.plotly_chart(fig, use_container_width=True)

        with lb:
            st.markdown("<div class='sec-header'>Statistik Deskriptif</div>", unsafe_allow_html=True)
            num_cols = df.select_dtypes('number').columns.tolist()
            desc = df[num_cols].describe().round(3)
            st.dataframe(desc.style.background_gradient(cmap='Blues', axis=0),
                         use_container_width=True)

        st.markdown("<div class='sec-header'>Detail Missing Values</div>", unsafe_allow_html=True)
        mv = pd.DataFrame({
            'Kolom': df.columns,
            'Missing': df.isnull().sum().values,
            'Persen (%)': (df.isnull().sum().values / len(df) * 100).round(2),
        }).query('Missing > 0')
        if mv.empty:
            st.success("Tidak ada missing values — dataset bersih.", icon=":material/check_circle:")
        else:
            st.dataframe(mv.reset_index(drop=True).style.background_gradient(
                cmap='Reds', subset=['Persen (%)']), use_container_width=True)

    else:
        st.markdown(f"""<div style='text-align:center;padding:64px 20px;
            border:1px dashed var(--line);color:var(--text-faint)'>
            {ic('folder', 22)}<br><span style='display:inline-block;margin-top:10px'>Upload file CSV terlebih dahulu</span>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# VISUALISASI
# ═══════════════════════════════════════════════════════════════════════════
elif page == 'viz':
    st.markdown("<div class='sec-header'>Visualisasi Data Interaktif</div>", unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("Upload dataset terlebih dahulu.", icon=":material/warning:")
    else:
        df  = st.session_state.df.copy()
        dfp = st.session_state.df_proc

        if 'Depression' in df.columns:
            df['Status'] = df['Depression'].map({0:'Tidak Depresi', 1:'Depresi'})

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Distribusi Depresi Berdasarkan Gender**")
            if 'Gender' in df.columns and 'Status' in df.columns:
                gdf = df.groupby(['Gender','Status']).size().reset_index(name='n')
                fig1 = px.bar(gdf, x='Gender', y='n', color='Status', barmode='group',
                              color_discrete_map=MONO, pattern_shape='Status',
                              pattern_shape_map=PATTERN, labels={'n':'Jumlah'})
                fig1.update_traces(marker_line_color='#000', marker_line_width=1, marker_pattern_fgcolor='#0a0a0a')
                fig1.update_layout(**PT, **AXIS, legend=dict(bgcolor='rgba(0,0,0,0)', font_color='#b3b3b3'))
                st.plotly_chart(fig1, use_container_width=True)

        with c2:
            st.markdown("**Tekanan Akademik vs Depresi**")
            if 'Academic Pressure' in df.columns and 'Status' in df.columns:
                fig2 = px.histogram(df, x='Academic Pressure', color='Status',
                                    nbins=10, barmode='overlay', opacity=.9,
                                    color_discrete_map=MONO, pattern_shape='Status',
                                    pattern_shape_map=PATTERN,
                                    labels={'Academic Pressure':'Tekanan Akademik','count':'Jumlah'})
                fig2.update_traces(marker_line_color='#000', marker_line_width=1, marker_pattern_fgcolor='#0a0a0a')
                fig2.update_layout(**PT, **AXIS, legend=dict(bgcolor='rgba(0,0,0,0)', font_color='#b3b3b3'))
                st.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            st.markdown("**Distribusi Durasi Tidur**")
            if 'Sleep Duration' in df.columns:
                order = ['Less than 5 hours','5-6 hours','7-8 hours','More than 8 hours']
                if 'Status' in df.columns:
                    sdf = df.groupby(['Sleep Duration','Status']).size().reset_index(name='n')
                    sdf['Sleep Duration'] = pd.Categorical(sdf['Sleep Duration'], categories=order, ordered=True)
                    sdf = sdf.sort_values('Sleep Duration')
                    fig3 = px.bar(sdf, x='Sleep Duration', y='n', color='Status',
                                  barmode='stack', color_discrete_map=MONO,
                                  pattern_shape='Status', pattern_shape_map=PATTERN,
                                  labels={'n':'Jumlah','Sleep Duration':'Durasi Tidur'})
                    fig3.update_traces(marker_line_color='#000', marker_line_width=1, marker_pattern_fgcolor='#0a0a0a')
                else:
                    sdf = df['Sleep Duration'].value_counts().reset_index()
                    sdf.columns = ['Sleep Duration', 'n']
                    fig3 = px.bar(sdf, x='Sleep Duration', y='n', color_discrete_sequence=['#e6e6e6'],
                                  labels={'n':'Jumlah','Sleep Duration':'Durasi Tidur'})
                fig3.update_layout(**PT, **AXIS, legend=dict(bgcolor='rgba(0,0,0,0)', font_color='#b3b3b3'))
                st.plotly_chart(fig3, use_container_width=True)

        with c4:
            sub_header("Heatmap Korelasi Fitur",
                       "Menunjukkan seberapa kuat hubungan antar fitur. Biru = berkorelasi "
                       "positif, merah = berkorelasi negatif. Makin pekat warnanya, makin "
                       "kuat hubungannya.")
            if dfp is not None:
                num_c = dfp.select_dtypes('number').columns.tolist()
                corr  = dfp[num_c].corr().round(2)
                fig4  = px.imshow(corr, text_auto=True, aspect='auto',
                                  color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
                fig4.update_layout(**PT, coloraxis_colorbar=dict(outlinewidth=0, tickfont_color='#7a7a7a'))
                st.plotly_chart(fig4, use_container_width=True)

        c5, c6 = st.columns(2)

        with c5:
            st.markdown("**Distribusi CGPA berdasarkan Status Depresi**")
            if 'CGPA' in df.columns and 'Status' in df.columns:
                fig5 = px.box(df, x='Status', y='CGPA', color='Status',
                              color_discrete_map=MONO, points='outliers')
                fig5.update_traces(marker_color='#8c8c8c', line_color='#e6e6e6')
                fig5.update_layout(**PT, **AXIS, showlegend=False)
                st.plotly_chart(fig5, use_container_width=True)

        with c6:
            st.markdown("**Stres Finansial vs Depresi (Scatter)**")
            if 'Financial Stress' in df.columns and 'CGPA' in df.columns and 'Status' in df.columns:
                fig6 = px.scatter(df.sample(min(500, len(df))),
                                  x='Financial Stress', y='CGPA',
                                  color='Status', color_discrete_map=MONO,
                                  symbol='Status', symbol_map={'Tidak Depresi':'circle','Depresi':'x'},
                                  opacity=.85,
                                  labels={'Financial Stress':'Stres Finansial'})
                fig6.update_layout(**PT, **AXIS, legend=dict(bgcolor='rgba(0,0,0,0)', font_color='#b3b3b3'))
                st.plotly_chart(fig6, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING MODEL
# ═══════════════════════════════════════════════════════════════════════════
elif page == 'train':
    st.markdown("<div class='sec-header'>Training Model Machine Learning</div>", unsafe_allow_html=True)

    if st.session_state.df_proc is None:
        st.warning("Upload dataset terlebih dahulu di halaman Data & Eksplorasi.", icon=":material/warning:")
    else:
        dfp = st.session_state.df_proc.copy()

        cfg, res = st.columns([1, 2])

        with cfg:
            st.markdown("<div class='sec-header'>Konfigurasi</div>", unsafe_allow_html=True)

            algo = st.selectbox("Algoritma", ["Decision Tree", "Naive Bayes"],
                help="Decision Tree: model pohon keputusan, mudah dibaca tapi bisa overfit "
                     "pada data kecil. Naive Bayes: model probabilistik, lebih cepat dan "
                     "stabil untuk data kategorikal.")
            test_pct = st.slider("Ukuran Data Test (%)", 10, 40, 20, 5,
                help="Persentase data yang disisihkan untuk menguji model setelah dilatih. "
                     "Sisanya dipakai untuk training.")

            if algo == "Decision Tree":
                max_d   = st.slider("Max Depth", 3, 20, 8, 1,
                    help="Batas kedalaman pohon keputusan. Makin dalam, makin detail tapi "
                         "makin rawan overfitting.")
                min_spl = st.slider("Min Samples Split", 2, 30, 4, 1,
                    help="Jumlah minimum data dalam satu node sebelum boleh dipecah lagi.")
                crit    = st.selectbox("Criterion", ["gini", "entropy"],
                    help="Ukuran yang dipakai pohon untuk memilih pemisahan terbaik: gini "
                         "(default, lebih cepat) atau entropy (berbasis information gain).")

            rand = st.number_input("Random State", 0, 9999, 115,
                help="Angka acuan pengacakan saat membagi data train/test. Nilai sama akan "
                     "menghasilkan pembagian yang sama setiap kali dijalankan.")
            go_train = st.button("Train Model", icon=":material/play_arrow:")

        with res:
            if go_train:
                if 'Depression' not in dfp.columns:
                    st.error("Kolom 'Depression' tidak ditemukan.", icon=":material/error:")
                else:
                    X = dfp.drop('Depression', axis=1)
                    y = dfp['Depression']

                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X, y, test_size=test_pct/100,
                        random_state=int(rand), stratify=y
                    )

                    with st.spinner("Melatih model..."):
                        if algo == "Decision Tree":
                            mdl = DecisionTreeClassifier(
                                max_depth=max_d, min_samples_split=min_spl,
                                criterion=crit, random_state=int(rand)
                            )
                        else:
                            mdl = GaussianNB()
                        mdl.fit(X_tr, y_tr)

                    y_tr_p = mdl.predict(X_tr)
                    y_te_p = mdl.predict(X_te)
                    tr_acc = accuracy_score(y_tr, y_tr_p)
                    te_acc = accuracy_score(y_te, y_te_p)

                    st.session_state.model       = mdl
                    st.session_state.model_name  = algo
                    st.session_state.feature_cols = X.columns.tolist()
                    st.session_state.train_acc   = tr_acc
                    st.session_state.test_acc    = te_acc
                    st.session_state.report      = classification_report(y_te, y_te_p, output_dict=True)
                    st.session_state.cm          = confusion_matrix(y_te, y_te_p)
                    if algo == "Decision Tree":
                        st.session_state.feat_imp = pd.DataFrame({
                            'Fitur': X.columns,
                            'Importance': mdl.feature_importances_,
                        }).sort_values('Importance', ascending=False)
                    else:
                        st.session_state.feat_imp = None

            if st.session_state.model is not None:
                tr_acc = st.session_state.train_acc
                te_acc = st.session_state.test_acc

                st.markdown("<div class='sec-header'>Hasil Training</div>", unsafe_allow_html=True)

                m1, m2 = st.columns(2)
                with m1:
                    ok   = tr_acc >= 0.80
                    note = f"{ic('check',13)} Target &ge;80% tercapai" if ok else f"{ic('x',13)} Di bawah target 80%"
                    st.markdown(f"""<div class='kpi-card'>
                        <div class='kpi-val'>{tr_acc*100:.2f}%</div>
                        <div class='kpi-label'>Akurasi Train</div>
                        <div style='color:var(--text-dim);font-size:.74rem;margin-top:6px'>{note}</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    ok   = te_acc >= 0.80
                    note = f"{ic('check',13)} Target &ge;80% tercapai" if ok else f"{ic('x',13)} Di bawah target 80%"
                    st.markdown(f"""<div class='kpi-card'>
                        <div class='kpi-val'>{te_acc*100:.2f}%</div>
                        <div class='kpi-label'>Akurasi Test</div>
                        <div style='color:var(--text-dim);font-size:.74rem;margin-top:6px'>{note}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                sub_header("Classification Report",
                           "<b style='color:var(--paper)'>precision</b>: dari semua yang diprediksi "
                           "positif, berapa persen benar.<br><b style='color:var(--paper)'>recall</b>: "
                           "dari semua yang sebenarnya positif, berapa persen tertangkap model.<br>"
                           "<b style='color:var(--paper)'>f1-score</b>: rata-rata seimbang antara "
                           "precision dan recall.")
                rpt_df = pd.DataFrame(st.session_state.report).T.round(4)
                rpt_df.index = rpt_df.index.map(
                    lambda x: 'Tidak Depresi' if x=='0' else ('Depresi' if x=='1' else x)
                )
                rpt_show = rpt_df.loc[[r for r in ['Tidak Depresi','Depresi','macro avg','weighted avg']
                                        if r in rpt_df.index]]
                metric_cols = [c for c in ['precision','recall','f1-score'] if c in rpt_show.columns]
                st.dataframe(rpt_show.style.background_gradient(cmap='Greens', subset=metric_cols),
                             use_container_width=True)

                sub_header("Confusion Matrix",
                           "Tabel yang membandingkan prediksi model dengan kondisi sebenarnya — "
                           "menunjukkan berapa banyak prediksi benar dan salah untuk tiap kelas.")
                cm  = st.session_state.cm
                cfig = px.imshow(
                    cm, text_auto=True, aspect='auto',
                    x=['Pred: Tidak Depresi','Pred: Depresi'],
                    y=['Actual: Tidak Depresi','Actual: Depresi'],
                    color_continuous_scale='Blues',
                )
                cfig.update_layout(**PT, coloraxis_showscale=False)
                cfig.update_traces(textfont_color='#0a0a0a')
                st.plotly_chart(cfig, use_container_width=True)

                fi = st.session_state.feat_imp
                if fi is not None:
                    sub_header("Feature Importance (Top 10)",
                               "Seberapa besar pengaruh tiap fitur terhadap keputusan model "
                               "Decision Tree. Makin tinggi nilainya, makin besar pengaruhnya "
                               "terhadap prediksi.")
                    fi_top = fi.head(10)
                    ffig = px.bar(fi_top, x='Importance', y='Fitur', orientation='h',
                                  color='Importance',
                                  color_continuous_scale=FEAT_SCALE)
                    ffig.update_layout(**PT,
                                       yaxis=dict(autorange='reversed', gridcolor=LINE),
                                       xaxis=dict(gridcolor=LINE),
                                       coloraxis_showscale=False)
                    st.plotly_chart(ffig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PREDIKSI
# ═══════════════════════════════════════════════════════════════════════════
elif page == 'predict':
    st.markdown("<div class='sec-header'>Prediksi Risiko Depresi Mahasiswa</div>", unsafe_allow_html=True)

    if st.session_state.model is None:
        st.warning("Latih model terlebih dahulu di halaman Training Model.", icon=":material/warning:")
    else:
        mdl   = st.session_state.model
        fcols = st.session_state.feature_cols
        dfp   = st.session_state.df_proc

        st.markdown(f"""<div class='box-info'>
            Model aktif: <b style='color:var(--paper)'>{st.session_state.model_name}</b> &nbsp;·&nbsp;
            Akurasi Test: <b style='color:var(--paper)'>{st.session_state.test_acc*100:.2f}%</b>
        </div><br>""", unsafe_allow_html=True)

        frm, out = st.columns([1, 1])

        with frm:
            st.markdown("<div class='sec-header'>Input Data Mahasiswa</div>", unsafe_allow_html=True)

            with st.form("pred_form"):
                f1, f2 = st.columns(2)
                with f1:
                    gender   = st.selectbox("Gender", ["Male", "Female"])
                    age      = st.number_input("Usia", 15, 60, 22)
                    cgpa     = st.number_input("CGPA", 0.0, 10.0, 7.5, 0.1,
                        help="Indeks Prestasi Kumulatif, skala 0 sampai 10. Menggambarkan "
                             "capaian akademik keseluruhan mahasiswa.")
                    sleep    = st.selectbox("Durasi Tidur", list(SLEEP_MAP.keys()), index=1,
                        help="Rata-rata durasi tidur per hari.")
                    diet     = st.selectbox("Kebiasaan Makan", list(DIET_MAP.keys()), index=1,
                        help="Pola makan sehari-hari: Unhealthy, Moderate, atau Healthy.")
                    degree   = st.selectbox("Jenjang Studi", ["BSc","BA","B.Tech","M.Tech","MBA","PhD","BCA","B.Pharm","Other"],
                        help="Jenjang pendidikan yang sedang atau telah ditempuh mahasiswa.")
                with f2:
                    acad_p   = st.slider("Tekanan Akademik", 1.0, 5.0, 3.0, 0.5,
                        help="Seberapa besar tekanan akademik yang dirasakan, dari 1 (rendah) "
                             "sampai 5 (sangat tinggi).")
                    work_p   = st.slider("Tekanan Kerja", 0.0, 5.0, 0.0, 0.5,
                        help="Tekanan dari pekerjaan sampingan. Isi 0 bila mahasiswa tidak bekerja.")
                    study_s  = st.slider("Kepuasan Belajar", 1.0, 5.0, 3.0, 0.5,
                        help="Tingkat kepuasan terhadap proses belajar, dari 1 (rendah) sampai 5 (tinggi).")
                    job_s    = st.slider("Kepuasan Kerja", 0.0, 5.0, 0.0, 0.5,
                        help="Tingkat kepuasan kerja. Isi 0 bila mahasiswa tidak bekerja.")
                    wsh      = st.number_input("Jam Belajar/Kerja per Hari", 0.0, 24.0, 6.0, 0.5,
                        help="Total jam belajar atau bekerja dalam satu hari.")
                    fin_s    = st.slider("Stres Finansial", 1.0, 5.0, 2.0, 0.5,
                        help="Tingkat tekanan finansial yang dirasakan, dari 1 (rendah) sampai "
                             "5 (sangat tinggi).")
                suicidal = st.selectbox("Pernah Punya Pikiran Bunuh Diri?", ["No", "Yes"],
                    help="Riwayat pernah memiliki pikiran untuk mengakhiri hidup. Data sensitif "
                         "ini dipakai murni sebagai salah satu variabel model.")
                fam_hist = st.selectbox("Riwayat Keluarga Sakit Mental", ["No", "Yes"],
                    help="Apakah ada anggota keluarga dengan riwayat gangguan kesehatan mental.")
                predict_btn = st.form_submit_button("Prediksi Sekarang", icon=":material/track_changes:")

            if predict_btn:
                inp = {
                    'Gender': BIN_MAP[gender],
                    'Age': float(age),
                    'Academic Pressure': acad_p,
                    'Work Pressure': work_p,
                    'CGPA': cgpa,
                    'Study Satisfaction': study_s,
                    'Job Satisfaction': job_s,
                    'Sleep Duration': SLEEP_MAP[sleep],
                    'Dietary Habits': DIET_MAP[diet],
                    'Have you ever had suicidal thoughts ?': BIN_MAP[suicidal],
                    'Work/Study Hours': wsh,
                    'Financial Stress': fin_s,
                    'Family History of Mental Illness': BIN_MAP[fam_hist],
                }
                for c in ['City', 'Degree']:
                    if c in fcols and dfp is not None and c in dfp.columns:
                        inp[c] = int(dfp[c].median())

                X_inp = np.array([[inp.get(c, 0) for c in fcols]])
                pred  = mdl.predict(X_inp)[0]
                proba = mdl.predict_proba(X_inp)[0]
                st.session_state.pred  = int(pred)
                st.session_state.proba = proba.tolist()

        with out:
            st.markdown("<div class='sec-header'>Hasil Prediksi</div>", unsafe_allow_html=True)

            if st.session_state.pred is not None:
                pred  = st.session_state.pred
                proba = st.session_state.proba
                p_dep = proba[1] * 100
                p_ok  = proba[0] * 100

                if pred == 1:
                    st.markdown(f"""<div class='box-danger'>
                        <div>{ic('alert', 36, 1.4)}</div>
                        <div style='font-size:1.3rem;font-weight:700;margin-top:10px;letter-spacing:.02em'>BERISIKO DEPRESI</div>
                        <div style='font-size:.84rem;margin-top:10px;line-height:1.65;color:#333'>
                            Mahasiswa ini menunjukkan indikator risiko depresi yang signifikan.
                            Disarankan untuk segera berkonsultasi dengan konselor atau psikolog.
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class='box-success'>
                        <div style='color:var(--paper)'>{ic('check', 36, 1.4)}</div>
                        <div style='font-size:1.3rem;font-weight:700;margin-top:10px;letter-spacing:.02em'>TIDAK BERISIKO</div>
                        <div style='color:var(--text-dim);font-size:.84rem;margin-top:10px;line-height:1.65'>
                            Mahasiswa ini tidak menunjukkan risiko depresi yang signifikan.
                            Tetap jaga kesehatan mental dengan pola hidup sehat.
                        </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>**Probabilitas**", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='prob-wrap'>
                    <div style='display:flex;justify-content:space-between'>
                        <span style='color:var(--text);font-size:.83rem'>Depresi</span>
                        <span style='color:#ff6b6b;font-weight:600;font-family:"IBM Plex Mono",monospace'>{p_dep:.1f}%</span>
                    </div>
                    <div class='prob-track'><div class='prob-fill' style='width:{p_dep}%;background:#ff6b6b'></div></div>
                </div>
                <div class='prob-wrap'>
                    <div style='display:flex;justify-content:space-between'>
                        <span style='color:var(--text);font-size:.83rem'>Tidak Depresi</span>
                        <span style='color:#5c7cfa;font-weight:600;font-family:"IBM Plex Mono",monospace'>{p_ok:.1f}%</span>
                    </div>
                    <div class='prob-track'><div class='prob-fill' style='width:{p_ok}%;background:#5c7cfa'></div></div>
                </div>
                """, unsafe_allow_html=True)

                # Gauge — risk zones colored (green -> amber -> red), bar reflects predicted class
                bar_color = '#ff6b6b' if pred == 1 else '#5c7cfa'
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=p_dep,
                    number={'suffix': '%', 'font': {'color': '#f2f2f2', 'size': 30, 'family': 'IBM Plex Mono'}},
                    title={'text': "Risiko Depresi", 'font': {'color': '#9a9a9a', 'size': 12}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': '#4d4d4d',
                                 'tickfont': {'color': '#5c5c5c', 'size': 10}},
                        'bar': {'color': bar_color, 'thickness': .3},
                        'bgcolor': '#131313',
                        'borderwidth': 1,
                        'bordercolor': '#2b2b2b',
                        'steps': [
                            {'range': [0,  40], 'color': 'rgba(81,207,102,.18)'},
                            {'range': [40, 65], 'color': 'rgba(255,212,59,.18)'},
                            {'range': [65,100], 'color': 'rgba(255,107,107,.20)'},
                        ],
                        'threshold': {
                            'line': {'color': '#f2f2f2', 'width': 2},
                            'thickness': .75, 'value': 50,
                        },
                    }
                ))
                gauge.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#e6e6e6',
                    height=220,
                    margin=dict(t=32, b=0, l=32, r=32),
                )
                st.plotly_chart(gauge, use_container_width=True)

            else:
                st.markdown(f"""<div style='text-align:center;padding:64px 20px;
                    border:1px dashed var(--line);color:var(--text-faint)'>
                    {ic('target', 22)}<br><span style='display:inline-block;margin-top:10px'>Isi form dan klik Prediksi Sekarang</span>
                </div>""", unsafe_allow_html=True)
