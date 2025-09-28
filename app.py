# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from pathlib import Path

# Page layout
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    base = Path(__file__).parent
    try:
        model = load(base / "model_knn.joblib")
    except Exception:
        st.error("Model file not found. Put model_knn.joblib in the app folder.")
        st.stop()
    try:
        scaler = load(base / "scaler.joblib")
    except Exception:
        st.error("Scaler file not found. Put scaler.joblib in the app folder.")
        st.stop()
    try:
        df = pd.read_csv(base / "heart.csv")
    except Exception:
        df = None
    return model, scaler, df

model, scaler, df = load_artifacts()

# ---------- CSS ----------
st.markdown(
    """
    <style>
    /* widen sidebar */
    section[data-testid="stSidebar"] { width: 420px !important; }

    /* align inputs */
    section[data-testid="stSidebar"] .stNumberInput,
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] .stTextInput,
    section[data-testid="stSidebar"] div[data-baseweb="input"] > div {
        width: 100% !important;
    }
    section[data-testid="stSidebar"] .css-1o0bg3o > div {
        padding-left: 6px;
        padding-right: 6px;
    }

    /* predict button style */
    section[data-testid="stSidebar"] div.stButton > button {
        width: 100%;
        height: 48px;
        font-size: 15px;
        border-radius: 8px;
    }

    /* result card: single unified container on right */
    .result-card {
        background: linear-gradient(180deg,#0f1720 0%, #08121a 100%);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.35);
        color: #e6eef6;
        min-height: 72vh; /* keep it tall so it looks like a full container */
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }

    .result-title { font-size: 1.1rem; margin-bottom: 8px; }
    .result-status { font-size: 1.05rem; margin-bottom: 6px; }
    .result-confidence { margin-top: 8px; }

    /* center main column */
    .main-centered { display:flex; justify-content:center; }

    /* chart grid spacing: two per row */
    .chart-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 24px;
        align-items: start;
    }

    /* responsive tweak for very small screens */
    @media (max-width: 900px) {
        .chart-grid { grid-template-columns: 1fr; }
        section[data-testid="stSidebar"] { width: 100% !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar: aligned inputs ----------
st.sidebar.header("Patient Input")

c1, c2 = st.sidebar.columns(2)
with c1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1)
with c2:
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x==0 else "Male", index=1)

c3, c4 = st.sidebar.columns(2)
with c3:
    cp = st.number_input("Chest pain type (cp)", min_value=0, max_value=3, value=0, step=1)
with c4:
    trestbps = st.number_input("Resting BP (trestbps)", min_value=50, max_value=250, value=130, step=1)

c5, c6 = st.sidebar.columns(2)
with c5:
    chol = st.number_input("Cholesterol (chol)", min_value=100, max_value=600, value=250, step=1)
with c6:
    fbs = st.selectbox("Fasting sugar >120 (fbs)", options=[0, 1], index=0)

c7, c8 = st.sidebar.columns(2)
with c7:
    restecg = st.number_input("Resting ECG (restecg)", min_value=0, max_value=2, value=0, step=1)
with c8:
    thalach = st.number_input("Max HR (thalach)", min_value=50, max_value=250, value=150, step=1)

c9, c10 = st.sidebar.columns(2)
with c9:
    exang = st.selectbox("Exercise angina (exang)", options=[0, 1], index=0)
with c10:
    oldpeak = st.number_input("ST depression (oldpeak)", value=1.0, format="%.2f")

c11, c12 = st.sidebar.columns(2)
with c11:
    slope = st.number_input("Slope (0–2)", min_value=0, max_value=2, value=1, step=1)
with c12:
    ca = st.number_input("Major vessels (ca)", min_value=0, max_value=4, value=0, step=1)

thal = st.sidebar.number_input("Thalassemia (thal)", min_value=0, max_value=3, value=2, step=1)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: provide sensible ranges for reliable predictions")

# ---------- Prediction stored in session_state ----------
def do_predict():
    x = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                   thalach, exang, oldpeak, slope, ca, thal]])
    try:
        x_scaled = scaler.transform(x)
        pred = int(model.predict(x_scaled)[0])
        prob = None
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(x_scaled)[0][pred])
        st.session_state["prediction"] = "Heart Disease" if pred == 1 else "No Heart Disease"
        st.session_state["confidence"] = (prob * 100) if prob is not None else None
    except Exception as e:
        st.session_state["prediction_error"] = str(e)

# Predict button in sidebar
st.sidebar.button("🔍 Predict", on_click=do_predict)

# ---------- Main layout ----------
st.title("Heart Disease Predictor (Streamlit)")

# Create columns: left spacer, main (charts), right (result container)
left_spacer, main_col, right_col = st.columns([0.2, 3.2, 1.0])

# Right: single container for result (one whole card like sidebar)
with right_col:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown('<div class="result-title">🩺 Prediction Result</div>', unsafe_allow_html=True)
    if "prediction_error" in st.session_state:
        st.error(f"Prediction error: {st.session_state.pop('prediction_error')}")
    elif "prediction" in st.session_state:
        p = st.session_state["prediction"]
        conf = st.session_state.get("confidence", None)
        # unified status line
        if p == "Heart Disease":
            st.markdown(f'<div class="result-status">⚠️ <strong style="color:#ffb4b4">{p}</strong></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-status">✅ <strong style="color:#9be7a2">{p}</strong></div>', unsafe_allow_html=True)
        if conf is not None:
            st.markdown(f'<div class="result-confidence">Confidence: <strong>{conf:.1f}%</strong></div>', unsafe_allow_html=True)
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.caption("Feature order: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal")
    else:
        st.info("No prediction yet. Use the sidebar inputs and click Predict.")
    st.markdown('</div>', unsafe_allow_html=True)

# Main: center charts; arrange in two-per-row grid with equal spacing
with main_col:
    st.markdown("## Exploratory Visuals")
    if df is not None:
        # Use a grid container (2 columns) with equal spacing
        st.markdown('<div class="chart-grid">', unsafe_allow_html=True)

        # Chart 1 (left)
        with st.container():
            st.write("Distribution of heart disease cases")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.countplot(x="target", data=df, palette="coolwarm", ax=ax1)
            ax1.set_xlabel("Heart Disease (1 = Yes, 0 = No)")
            st.pyplot(fig1, clear_figure=True)

        # Chart 2 (right)
        with st.container():
            st.write("Age vs Cholesterol")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x="age", y="chol", hue="target", data=df, palette="coolwarm", ax=ax2)
            st.pyplot(fig2, clear_figure=True)

        # Chart 3 (left - second row)
        with st.container():
            st.write("Cholesterol by Heart Disease (boxplot)")
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.boxplot(x="target", y="chol", data=df, palette="Set2", ax=ax3)
            st.pyplot(fig3, clear_figure=True)

        # Chart 4 (right - second row)
        with st.container():
            st.write("Resting blood pressure distribution")
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            sns.histplot(df["trestbps"], bins=30, kde=True, ax=ax4)
            st.pyplot(fig4, clear_figure=True)

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Dataset `heart.csv` not found — upload it to enable visuals.")
