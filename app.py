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

    /* align sidebar inputs */
    section[data-testid="stSidebar"] .css-1o0bg3o > div { padding-left:8px; padding-right:8px; }
    section[data-testid="stSidebar"] .stNumberInput, section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] div[data-baseweb="input"] > div { width:100% !important; }

    /* full-width predict button */
    section[data-testid="stSidebar"] div.stButton > button {
        width:100%; height:46px; font-size:15px; border-radius:8px;
    }

    /* result card styling + make card a fixed min-height so it looks like one single container */
    .result-card {
        background: linear-gradient(180deg,#0f1720 0%, #091018 100%);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.28);
        color: #e6eef6;
        min-height: 520px; /* controls visual height of the card */
    }
    .result-card h3 { margin: 0 0 8px 0; }
    .result-status { font-size:1.05rem; margin-bottom:6px; }
    .result-confidence { margin-top:10px; }

    /* center main column and give it max width so charts don't span too wide */
    .main-centered {
        display:flex;
        justify-content:center;
    }
    .charts-wrapper {
        width: 100%;
        max-width: 980px; /* center column width */
    }

    /* small spacing for chart tiles */
    .chart-tile {
        padding: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar inputs ----------
st.sidebar.header("Patient Input")

c1, c2 = st.sidebar.columns(2)
with c1:
    age = st.number_input("Age", 1, 120, 50, step=1)
with c2:
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", index=1)

c3, c4 = st.sidebar.columns(2)
with c3:
    cp = st.number_input("Chest pain type (cp)", 0, 3, 0, step=1)
with c4:
    trestbps = st.number_input("Resting BP (trestbps)", 50, 250, 130, step=1)

c5, c6 = st.sidebar.columns(2)
with c5:
    chol = st.number_input("Cholesterol (chol)", 100, 600, 250, step=1)
with c6:
    fbs = st.selectbox("Fasting sugar >120 (fbs)", options=[0, 1], index=0)

c7, c8 = st.sidebar.columns(2)
with c7:
    restecg = st.number_input("Resting ECG (restecg)", 0, 2, 0, step=1)
with c8:
    thalach = st.number_input("Max HR (thalach)", 50, 250, 150, step=1)

c9, c10 = st.sidebar.columns(2)
with c9:
    exang = st.selectbox("Exercise angina (exang)", options=[0, 1], index=0)
with c10:
    oldpeak = st.number_input("ST depression (oldpeak)", value=1.0, format="%.2f")

c11, c12 = st.sidebar.columns(2)
with c11:
    slope = st.number_input("Slope (0–2)", 0, 2, 1, step=1)
with c12:
    ca = st.number_input("Major vessels (ca)", 0, 4, 0, step=1)

thal = st.sidebar.number_input("Thalassemia (thal)", 0, 3, 2, step=1)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: provide sensible ranges for reliable predictions")

# ---------- Prediction logic (session_state) ----------
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

st.sidebar.button("🔍 Predict", on_click=do_predict)

# ---------- Main layout ----------
st.title("Heart Disease Predictor (Streamlit)")

# Create 3 columns: left spacer, main (centered), right result card
left_spacer, main_col, right_col = st.columns([0.2, 2.6, 0.9])

# Right: unified result container (one single card)
with right_col:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown("<h3>🩺 Prediction Result</h3>", unsafe_allow_html=True)
    if "prediction_error" in st.session_state:
        st.error(f"Prediction error: {st.session_state.pop('prediction_error')}")
    elif "prediction" in st.session_state:
        p = st.session_state["prediction"]
        conf = st.session_state.get("confidence", None)
        if p == "Heart Disease":
            st.markdown(f'<div class="result-status"><strong style="color:#ffb4b4">⚠️ {p}</strong></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-status"><strong style="color:#9be7a2">✅ {p}</strong></div>', unsafe_allow_html=True)
        if conf is not None:
            st.markdown(f'<div class="result-confidence">Confidence: <strong>{conf:.1f}%</strong></div>', unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.caption("Feature order: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal")
    else:
        st.info("No prediction yet. Use the sidebar inputs and click Predict.")
    st.markdown('</div>', unsafe_allow_html=True)

# Main: center charts in a 2x2 grid with margins
with main_col:
    st.markdown('<div class="main-centered">', unsafe_allow_html=True)
    st.markdown('<div class="charts-wrapper">', unsafe_allow_html=True)

    st.markdown("## Exploratory Visuals")
    st.markdown('<div class="scroll-container">', unsafe_allow_html=True)

    if df is not None:
        # First row: two charts side-by-side
        r1c1, r1c2 = st.columns([1,1], gap="large")
        with r1c1:
            st.markdown('<div class="chart-tile">', unsafe_allow_html=True)
            st.write("Distribution of heart disease cases")
            fig1, ax1 = plt.subplots(figsize=(5.5, 3.6))
            sns.countplot(x="target", data=df, palette="coolwarm", ax=ax1)
            ax1.set_xlabel("Heart Disease (1 = Yes, 0 = No)")
            st.pyplot(fig1, clear_figure=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with r1c2:
            st.markdown('<div class="chart-tile">', unsafe_allow_html=True)
            st.write("Age vs Cholesterol")
            fig2, ax2 = plt.subplots(figsize=(5.5, 3.6))
            sns.scatterplot(x="age", y="chol", hue="target", data=df, palette="coolwarm", ax=ax2)
            st.pyplot(fig2, clear_figure=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Second row: two charts side-by-side
        r2c1, r2c2 = st.columns([1,1], gap="large")
        with r2c1:
            st.markdown('<div class="chart-tile">', unsafe_allow_html=True)
            st.write("Cholesterol by Heart Disease (boxplot)")
            fig3, ax3 = plt.subplots(figsize=(5.5, 3.6))
            sns.boxplot(x="target", y="chol", data=df, palette="Set2", ax=ax3)
            st.pyplot(fig3, clear_figure=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with r2c2:
            st.markdown('<div class="chart-tile">', unsafe_allow_html=True)
            st.write("Resting BP distribution")
            fig4, ax4 = plt.subplots(figsize=(5.5, 3.6))
            sns.histplot(df["trestbps"], bins=30, kde=True, ax=ax4)
            st.pyplot(fig4, clear_figure=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Dataset `heart.csv` not found — upload it to enable visuals.")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
