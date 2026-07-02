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
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #0a0f1e;
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1a2544 100%);
    border-right: 1px solid #1e293b;
}

/* hide hamburger & footer */
#MainMenu, footer, header { visibility: hidden; }

/* Metric cards */
.kpi-card {
    background: linear-gradient(135deg, #1e293b 0%, #162032 100%);
    border: 1px solid #2d3f5e;
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
    transition: transform .2s, box-shadow .2s;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #3b82f6, #6366f1);
    border-radius: 14px 14px 0 0;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(59,130,246,.18);
}
.kpi-val {
    font-size: 2.1rem;
    font-weight: 800;
    color: #60a5fa;
    line-height: 1;
}
.kpi-label {
    font-size: .72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: .1em;
    margin-top: 6px;
    font-weight: 600;
}

/* Section headers */
.sec-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #f1f5f9;
    border-left: 3px solid #3b82f6;
    padding-left: 12px;
    margin: 28px 0 16px;
}

/* Info / alert boxes */
.box-info {
    background: rgba(59,130,246,.1);
    border: 1px solid rgba(59,130,246,.3);
    border-radius: 10px;
    padding: 14px 18px;
    color: #93c5fd;
    font-size: .88rem;
}
.box-danger {
    background: rgba(239,68,68,.12);
    border: 1px solid rgba(239,68,68,.35);
    border-radius: 12px;
    padding: 24px 20px;
    text-align: center;
}
.box-success {
    background: rgba(34,197,94,.1);
    border: 1px solid rgba(34,197,94,.35);
    border-radius: 12px;
    padding: 24px 20px;
    text-align: center;
}

/* Prediction probability bars */
.prob-wrap { margin: 6px 0 12px; }
.prob-track {
    background: #1e293b;
    border-radius: 100px;
    height: 10px;
    overflow: hidden;
    margin-top: 4px;
}
.prob-fill-red  { height: 100%; border-radius: 100px; background: linear-gradient(90deg,#ef4444,#dc2626); }
.prob-fill-green{ height: 100%; border-radius: 100px; background: linear-gradient(90deg,#22c55e,#16a34a); }

/* Step cards */
.step-card {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    background: #1e293b;
    border: 1px solid #2d3f5e;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg,#3b82f6,#2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all .2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#60a5fa,#3b82f6) !important;
    box-shadow: 0 4px 20px rgba(59,130,246,.4) !important;
    transform: translateY(-1px) !important;
}

/* Form inputs */
.stSelectbox label, .stSlider label, .stNumberInput label {
    color: #94a3b8 !important;
    font-size: .82rem !important;
    font-weight: 500 !important;
}

/* Streamlit widgets dark override */
.stSelectbox > div > div {
    background-color: #1e293b !important;
    border-color: #334155 !important;
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ────────────────────────────────────────────────────────────
def _ss():
    defaults = dict(
        df=None, df_proc=None, le={},
        model=None, model_name=None, feature_cols=None,
        train_acc=None, test_acc=None,
        report=None, cm=None, feat_imp=None,
        pred=None, proba=None,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_ss()


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

    # fill missing
    for c in df.select_dtypes('number').columns:
        df[c] = df[c].fillna(df[c].median())
    for c in df.select_dtypes('object').columns:
        df[c] = df[c].fillna(df[c].mode()[0])

    le_store = {}

    # binary
    for c in ['Gender', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']:
        if c in df.columns:
            df[c] = df[c].map(BIN_MAP).fillna(0).astype(int)

    # ordinal
    if 'Sleep Duration' in df.columns:
        df['Sleep Duration'] = df['Sleep Duration'].map(SLEEP_MAP).fillna(2).astype(int)
    if 'Dietary Habits'  in df.columns:
        df['Dietary Habits'] = df['Dietary Habits'].map(DIET_MAP).fillna(1).astype(int)

    # label-encode remaining categoricals
    for c in ['City', 'Degree']:
        if c in df.columns:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            le_store[c] = le

    return df, le_store


# ── Plot theme ───────────────────────────────────────────────────────────────
PT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(14,20,36,.85)',
    font_color='#cbd5e1',
    title_font_color='#f1f5f9',
    margin=dict(t=24, b=24, l=24, r=24),
)
AXIS = dict(
    xaxis=dict(gridcolor='#1e293b', linecolor='#334155', tickfont_color='#64748b'),
    yaxis=dict(gridcolor='#1e293b', linecolor='#334155', tickfont_color='#64748b'),
)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:24px 0 12px'>
        <div style='font-size:2.4rem'>🧠</div>
        <div style='font-size:1.15rem;font-weight:800;color:#f1f5f9;margin-top:6px'>MindScope</div>
        <div style='font-size:.72rem;color:#475569;margin-top:2px;letter-spacing:.08em'>STUDENT DEPRESSION AI</div>
    </div>
    <hr style='border-color:#1e293b;margin-bottom:20px'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["🏠  Dashboard", "📂  Data & Eksplorasi", "📊  Visualisasi",
         "🤖  Training Model", "🔮  Prediksi"],
        label_visibility="collapsed",
    )

    st.markdown("""<hr style='border-color:#1e293b;margin-top:20px'>
    <div style='font-size:.68rem;color:#334155;text-align:center;padding:8px 0'>
        Data Mining Project · 2024<br>
        Kaggle: Student Depression Dataset
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HOME / DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Dashboard":
    st.markdown("""
    <div style='padding:52px 0 32px;text-align:center'>
        <div style='font-size:.78rem;font-weight:700;color:#3b82f6;letter-spacing:.16em;
                    text-transform:uppercase;margin-bottom:14px'>Data Mining Project</div>
        <h1 style='font-size:3rem;font-weight:800;margin:0;line-height:1.15;
            background:linear-gradient(135deg,#f1f5f9 30%,#3b82f6);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
            Student Depression<br>Risk Predictor
        </h1>
        <p style='color:#64748b;margin-top:18px;font-size:.95rem;max-width:560px;
                  margin-left:auto;margin-right:auto;line-height:1.7'>
            Prediksi risiko depresi mahasiswa menggunakan algoritma Machine Learning
            berbasis data psikologis, akademik, dan gaya hidup.
        </p>
    </div>
    """, unsafe_allow_html=True)

    df = st.session_state.df
    c1, c2, c3, c4 = st.columns(4)
    vals = [
        (len(df) if df is not None else "—", "Total Data"),
        ((len(df.columns) - 2) if df is not None else "—", "Jumlah Fitur"),
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
            Dua algoritma tersedia: <b>Decision Tree</b> (dapat diinterpretasi, feature importance)
            dan <b>Naive Bayes</b> (probabilistik, cepat). Pilih dan bandingkan keduanya di halaman
            Training Model.
        </div>""", unsafe_allow_html=True)

    with lb:
        st.markdown("<div class='sec-header'>Alur Penggunaan</div>", unsafe_allow_html=True)
        for icon, title, desc in [
            ("📂", "Upload Dataset", "Unggah CSV di halaman Data & Eksplorasi"),
            ("📊", "Eksplorasi & Visualisasi", "Lihat distribusi dan korelasi fitur"),
            ("🤖", "Training Model", "Pilih algoritma, atur parameter, latih"),
            ("🔮", "Prediksi", "Masukkan profil mahasiswa → hasil instan"),
        ]:
            st.markdown(f"""<div class='step-card'>
                <span style='font-size:1.3rem;flex-shrink:0'>{icon}</span>
                <div>
                    <div style='font-weight:600;color:#f1f5f9;font-size:.88rem'>{title}</div>
                    <div style='color:#475569;font-size:.78rem;margin-top:2px'>{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    if df is None:
        st.markdown("""<br><div style='text-align:center;padding:20px;
            background:rgba(59,130,246,.06);border:1px dashed #3b82f6;
            border-radius:12px;color:#60a5fa;font-size:.88rem'>
            👆 Mulai dengan upload dataset di halaman <b>Data & Eksplorasi</b>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA & EKSPLORASI
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📂  Data & Eksplorasi":
    st.markdown("<div class='sec-header'>Upload & Eksplorasi Dataset</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload file CSV", type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state.df = df
        df_proc, le = preprocess(df)
        st.session_state.df_proc = df_proc
        st.session_state.le = le
        st.session_state.model = None      # reset model on new data
        st.success(f"✅ Dataset dimuat: **{len(df):,} baris** · **{len(df.columns)} kolom**")

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
            (depresi_pct, "% Depresi"),
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
                             color_discrete_sequence=['#3b82f6','#ef4444'], hole=.4)
                fig.update_layout(**PT, legend=dict(bgcolor='rgba(0,0,0,0)'))
                st.plotly_chart(fig, use_container_width=True)

        with lb:
            st.markdown("<div class='sec-header'>Statistik Deskriptif</div>", unsafe_allow_html=True)
            num_cols = df.select_dtypes('number').columns.tolist()
            st.dataframe(df[num_cols].describe().round(3), use_container_width=True)

        # Missing values table
        st.markdown("<div class='sec-header'>Detail Missing Values</div>", unsafe_allow_html=True)
        mv = pd.DataFrame({
            'Kolom': df.columns,
            'Missing': df.isnull().sum().values,
            'Persen (%)': (df.isnull().sum().values / len(df) * 100).round(2),
        }).query('Missing > 0')
        if mv.empty:
            st.success("✅ Tidak ada missing values — dataset bersih!")
        else:
            st.dataframe(mv.reset_index(drop=True), use_container_width=True)

    else:
        st.markdown("""<div style='text-align:center;padding:60px 20px;
            background:rgba(30,41,59,.5);border:1px dashed #334155;
            border-radius:12px;color:#475569'>
            📁 Upload file CSV terlebih dahulu
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISASI
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Visualisasi":
    st.markdown("<div class='sec-header'>Visualisasi Data Interaktif</div>", unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Upload dataset terlebih dahulu.")
    else:
        df  = st.session_state.df.copy()
        dfp = st.session_state.df_proc

        if 'Depression' in df.columns:
            df['Status'] = df['Depression'].map({0:'Tidak Depresi', 1:'Depresi'})

        COLOR = {'Tidak Depresi':'#3b82f6', 'Depresi':'#ef4444'}

        # Row 1
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Distribusi Depresi Berdasarkan Gender**")
            if 'Gender' in df.columns and 'Status' in df.columns:
                gdf = df.groupby(['Gender','Status']).size().reset_index(name='n')
                fig1 = px.bar(gdf, x='Gender', y='n', color='Status', barmode='group',
                              color_discrete_map=COLOR, labels={'n':'Jumlah'})
                fig1.update_layout(**PT, **AXIS, legend=dict(bgcolor='rgba(0,0,0,0)'))
                st.plotly_chart(fig1, use_container_width=True)

        with c2:
            st.markdown("**Tekanan Akademik vs Depresi**")
            if 'Academic Pressure' in df.columns and 'Status' in df.columns:
                fig2 = px.histogram(df, x='Academic Pressure', color='Status',
                                    nbins=10, barmode='overlay', opacity=.75,
                                    color_discrete_map=COLOR,
                                    labels={'Academic Pressure':'Tekanan Akademik','count':'Jumlah'})
                fig2.update_layout(**PT, **AXIS, legend=dict(bgcolor='rgba(0,0,0,0)'))
                st.plotly_chart(fig2, use_container_width=True)

        # Row 2
        c3, c4 = st.columns(2)

        with c3:
            st.markdown("**Distribusi Durasi Tidur**")
            if 'Sleep Duration' in df.columns:
                order = ['Less than 5 hours','5-6 hours','7-8 hours','More than 8 hours']
                sdf = (df.groupby(['Sleep Duration','Status']).size()
                         .reset_index(name='n') if 'Status' in df.columns
                       else df['Sleep Duration'].value_counts().reset_index().rename(columns={'index':'Sleep Duration','Sleep Duration':'n'}))
                if 'Status' in sdf.columns:
                    sdf['Sleep Duration'] = pd.Categorical(sdf['Sleep Duration'], categories=order, ordered=True)
                    sdf = sdf.sort_values('Sleep Duration')
                    fig3 = px.bar(sdf, x='Sleep Duration', y='n', color='Status',
                                  barmode='stack', color_discrete_map=COLOR,
                                  labels={'n':'Jumlah','Sleep Duration':'Durasi Tidur'})
                else:
                    fig3 = px.bar(sdf, x='Sleep Duration', y='n',
                                  labels={'n':'Jumlah','Sleep Duration':'Durasi Tidur'})
                fig3.update_layout(**PT, **AXIS, legend=dict(bgcolor='rgba(0,0,0,0)'))
                st.plotly_chart(fig3, use_container_width=True)

        with c4:
            st.markdown("**Heatmap Korelasi Fitur**")
            if dfp is not None:
                num_c = dfp.select_dtypes('number').columns.tolist()
                corr  = dfp[num_c].corr().round(2)
                fig4  = px.imshow(corr, text_auto=True, aspect='auto',
                                  color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
                fig4.update_layout(**PT)
                st.plotly_chart(fig4, use_container_width=True)

        # Row 3: bonus
        c5, c6 = st.columns(2)

        with c5:
            st.markdown("**Distribusi CGPA berdasarkan Status Depresi**")
            if 'CGPA' in df.columns and 'Status' in df.columns:
                fig5 = px.box(df, x='Status', y='CGPA', color='Status',
                              color_discrete_map=COLOR, points='outliers')
                fig5.update_layout(**PT, **AXIS, showlegend=False)
                st.plotly_chart(fig5, use_container_width=True)

        with c6:
            st.markdown("**Stres Finansial vs Depresi (Scatter)**")
            if 'Financial Stress' in df.columns and 'CGPA' in df.columns and 'Status' in df.columns:
                fig6 = px.scatter(df.sample(min(500, len(df))),
                                  x='Financial Stress', y='CGPA',
                                  color='Status', color_discrete_map=COLOR,
                                  opacity=.7,
                                  labels={'Financial Stress':'Stres Finansial'})
                fig6.update_layout(**PT, **AXIS, legend=dict(bgcolor='rgba(0,0,0,0)'))
                st.plotly_chart(fig6, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING MODEL
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Training Model":
    st.markdown("<div class='sec-header'>Training Model Machine Learning</div>", unsafe_allow_html=True)

    if st.session_state.df_proc is None:
        st.warning("⚠️ Upload dataset terlebih dahulu di halaman Data & Eksplorasi.")
    else:
        dfp = st.session_state.df_proc.copy()

        cfg, res = st.columns([1, 2])

        with cfg:
            st.markdown("<div class='sec-header'>Konfigurasi</div>", unsafe_allow_html=True)

            algo = st.selectbox("Algoritma", ["Decision Tree", "Naive Bayes"])
            test_pct = st.slider("Ukuran Data Test (%)", 10, 40, 20, 5)

            if algo == "Decision Tree":
                max_d   = st.slider("Max Depth", 3, 20, 8, 1)
                min_spl = st.slider("Min Samples Split", 2, 30, 4, 1)
                crit    = st.selectbox("Criterion", ["gini", "entropy"])

            rand = st.number_input("Random State", 0, 9999, 115)
            go_train = st.button("🚀 Train Model")

        with res:
            if go_train:
                if 'Depression' not in dfp.columns:
                    st.error("Kolom 'Depression' tidak ditemukan!")
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

            # Render persisted results
            if st.session_state.model is not None:
                tr_acc = st.session_state.train_acc
                te_acc = st.session_state.test_acc

                st.markdown("<div class='sec-header'>Hasil Training</div>", unsafe_allow_html=True)

                m1, m2 = st.columns(2)
                with m1:
                    ok   = tr_acc >= 0.80
                    col  = '#22c55e' if ok else '#ef4444'
                    note = '✅ Target ≥80% tercapai' if ok else '❌ Di bawah target 80%'
                    st.markdown(f"""<div class='kpi-card'>
                        <div class='kpi-val' style='color:{col}'>{tr_acc*100:.2f}%</div>
                        <div class='kpi-label'>Akurasi Train</div>
                        <div style='color:{col};font-size:.75rem;margin-top:4px'>{note}</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    ok   = te_acc >= 0.80
                    col  = '#22c55e' if ok else '#ef4444'
                    note = '✅ Target ≥80% tercapai' if ok else '❌ Di bawah target 80%'
                    st.markdown(f"""<div class='kpi-card'>
                        <div class='kpi-val' style='color:{col}'>{te_acc*100:.2f}%</div>
                        <div class='kpi-label'>Akurasi Test</div>
                        <div style='color:{col};font-size:.75rem;margin-top:4px'>{note}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Classification report
                st.markdown("**Classification Report:**")
                rpt_df = pd.DataFrame(st.session_state.report).T.round(4)
                # keep only class rows + weighted avg
                display_rows = [r for r in rpt_df.index if r not in ('accuracy',)]
                rpt_df.index = rpt_df.index.map(
                    lambda x: 'Tidak Depresi' if x=='0' else ('Depresi' if x=='1' else x)
                )
                st.dataframe(rpt_df.loc[[r for r in ['Tidak Depresi','Depresi','macro avg','weighted avg']
                                          if r in rpt_df.index]],
                             use_container_width=True)

                # Confusion matrix
                st.markdown("**Confusion Matrix:**")
                cm  = st.session_state.cm
                cfig = px.imshow(
                    cm, text_auto=True, aspect='auto',
                    x=['Pred: Tidak Depresi','Pred: Depresi'],
                    y=['Actual: Tidak Depresi','Actual: Depresi'],
                    color_continuous_scale='Blues',
                )
                cfig.update_layout(**PT)
                st.plotly_chart(cfig, use_container_width=True)

                # Feature importance
                fi = st.session_state.feat_imp
                if fi is not None:
                    st.markdown("**Feature Importance (Top 10):**")
                    fi_top = fi.head(10)
                    ffig = px.bar(fi_top, x='Importance', y='Fitur', orientation='h',
                                  color='Importance',
                                  color_continuous_scale=['#1e293b','#3b82f6','#60a5fa'])
                    ffig.update_layout(**PT,
                                       yaxis=dict(autorange='reversed', gridcolor='#1e293b'),
                                       xaxis=dict(gridcolor='#1e293b'),
                                       coloraxis_showscale=False)
                    st.plotly_chart(ffig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PREDIKSI
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Prediksi":
    st.markdown("<div class='sec-header'>Prediksi Risiko Depresi Mahasiswa</div>", unsafe_allow_html=True)

    if st.session_state.model is None:
        st.warning("⚠️ Latih model terlebih dahulu di halaman Training Model.")
    else:
        mdl   = st.session_state.model
        fcols = st.session_state.feature_cols
        dfp   = st.session_state.df_proc

        st.markdown(f"""<div class='box-info'>
            Model aktif: <b>{st.session_state.model_name}</b> &nbsp;|&nbsp;
            Akurasi Test: <b>{st.session_state.test_acc*100:.2f}%</b>
        </div><br>""", unsafe_allow_html=True)

        frm, out = st.columns([1, 1])

        with frm:
            st.markdown("<div class='sec-header'>Input Data Mahasiswa</div>", unsafe_allow_html=True)

            with st.form("pred_form"):
                f1, f2 = st.columns(2)
                with f1:
                    gender   = st.selectbox("Gender", ["Male", "Female"])
                    age      = st.number_input("Usia", 15, 60, 22)
                    cgpa     = st.number_input("CGPA", 0.0, 10.0, 7.5, 0.1)
                    sleep    = st.selectbox("Durasi Tidur", list(SLEEP_MAP.keys()), index=1)
                    diet     = st.selectbox("Kebiasaan Makan", list(DIET_MAP.keys()), index=1)
                    degree   = st.selectbox("Jenjang Studi", ["BSc","BA","B.Tech","M.Tech","MBA","PhD","BCA","B.Pharm","Other"])
                with f2:
                    acad_p   = st.slider("Tekanan Akademik", 1.0, 5.0, 3.0, 0.5)
                    work_p   = st.slider("Tekanan Kerja", 0.0, 5.0, 0.0, 0.5)
                    study_s  = st.slider("Kepuasan Belajar", 1.0, 5.0, 3.0, 0.5)
                    job_s    = st.slider("Kepuasan Kerja", 0.0, 5.0, 0.0, 0.5)
                    wsh      = st.number_input("Jam Belajar/Kerja per Hari", 0.0, 24.0, 6.0, 0.5)
                    fin_s    = st.slider("Stres Finansial", 1.0, 5.0, 2.0, 0.5)
                suicidal = st.selectbox("Pernah Punya Pikiran Bunuh Diri?", ["No", "Yes"])
                fam_hist = st.selectbox("Riwayat Keluarga Sakit Mental", ["No", "Yes"])
                predict_btn = st.form_submit_button("🔮 Prediksi Sekarang")

            if predict_btn:
                # build input dict
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
                # fill label-encoded cols with median of training data
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
                        <div style='font-size:2.8rem;margin-bottom:8px'>⚠️</div>
                        <div style='font-size:1.35rem;font-weight:800;color:#ef4444'>BERISIKO DEPRESI</div>
                        <div style='color:#fca5a5;font-size:.84rem;margin-top:10px;line-height:1.6'>
                            Mahasiswa ini menunjukkan indikator risiko depresi yang signifikan.
                            Disarankan untuk segera berkonsultasi dengan konselor atau psikolog.
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class='box-success'>
                        <div style='font-size:2.8rem;margin-bottom:8px'>✅</div>
                        <div style='font-size:1.35rem;font-weight:800;color:#22c55e'>TIDAK BERISIKO</div>
                        <div style='color:#86efac;font-size:.84rem;margin-top:10px;line-height:1.6'>
                            Mahasiswa ini tidak menunjukkan risiko depresi yang signifikan.
                            Tetap jaga kesehatan mental dengan pola hidup sehat.
                        </div>
                    </div>""", unsafe_allow_html=True)

                # Probability bars
                st.markdown("<br>**Probabilitas:**", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='prob-wrap'>
                    <div style='display:flex;justify-content:space-between'>
                        <span style='color:#ef4444;font-size:.83rem'>🔴 Depresi</span>
                        <span style='color:#ef4444;font-weight:700'>{p_dep:.1f}%</span>
                    </div>
                    <div class='prob-track'><div class='prob-fill-red' style='width:{p_dep}%'></div></div>
                </div>
                <div class='prob-wrap'>
                    <div style='display:flex;justify-content:space-between'>
                        <span style='color:#22c55e;font-size:.83rem'>🟢 Tidak Depresi</span>
                        <span style='color:#22c55e;font-weight:700'>{p_ok:.1f}%</span>
                    </div>
                    <div class='prob-track'><div class='prob-fill-green' style='width:{p_ok}%'></div></div>
                </div>
                """, unsafe_allow_html=True)

                # Gauge chart
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=p_dep,
                    number={'suffix': '%', 'font': {'color': '#e2e8f0', 'size': 30}},
                    title={'text': "Risiko Depresi", 'font': {'color': '#94a3b8', 'size': 13}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': '#475569',
                                 'tickfont': {'color': '#475569', 'size': 10}},
                        'bar': {'color': '#ef4444' if pred == 1 else '#22c55e', 'thickness': .3},
                        'bgcolor': '#1e293b',
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0,  40], 'color': 'rgba(34,197,94,.12)'},
                            {'range': [40, 65], 'color': 'rgba(251,191,36,.12)'},
                            {'range': [65,100], 'color': 'rgba(239,68,68,.12)'},
                        ],
                        'threshold': {
                            'line': {'color': '#f1f5f9', 'width': 2},
                            'thickness': .75, 'value': 50,
                        },
                    }
                ))
                gauge.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#e2e8f0',
                    height=230,
                    margin=dict(t=32, b=0, l=32, r=32),
                )
                st.plotly_chart(gauge, use_container_width=True)

            else:
                st.markdown("""<div style='text-align:center;padding:60px 20px;
                    background:rgba(30,41,59,.5);border:1px dashed #334155;
                    border-radius:12px;color:#475569'>
                    🔮 Isi form dan klik Prediksi Sekarang
                </div>""", unsafe_allow_html=True)
