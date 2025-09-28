# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from pathlib import Path

# ---------- Page config ----------
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    model = None
    scaler = None
    df = None
    base = Path(__file__).parent
    try:
        model = load(base / "model_knn.joblib")
    except Exception:
        st.error("Model file not found. Upload `model_knn.joblib`.")
        st.stop()
    try:
        scaler = load(base / "scaler.joblib")
    except Exception:
        st.error("Scaler file not found. Upload `scaler.joblib`.")
        st.stop()
    try:
        df = pd.read_csv(base / "heart.csv")
    except Exception:
        df = None
    return model, scaler, df

model, scaler, df = load_artifacts()

# ---------- Custom CSS ----------
st.markdown(
    """
    <style>
    /* Sidebar wider */
    section[data-testid="stSidebar"] {
        width: 420px !important;
    }

    /* Button style */
    div.stButton > button {
        width: 100%;
        height: 50px;
        font-size: 16px;
        border-radius: 10px;
    }

    /* Scroll area for charts */
    .scroll-container {
        max-height: 650px;
        overflow-y: auto;
        padding-right: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
st.sidebar.header("Patient Input")

# Grouped inputs: 2 per row
c1, c2 = st.sidebar.columns(2)
with c1:
    age = st.number_input("Age", 1, 120, 50, step=1)
with c2:
    sex = st.selectbox("Sex (0=female, 1=male)", [0, 1], index=1)

c3, c4 = st.sidebar.columns(2)
with c3:
    cp = st.number_input("Chest pain type (cp)", 0, 3, 0, step=1)
with c4:
    trestbps = st.number_input("Resting BP (trestbps)", 50, 250, 130, step=1)

c5, c6 = st.sidebar.columns(2)
with c5:
    chol = st.number_input("Cholesterol (chol)", 100, 600, 250, step=1)
with c6:
    fbs = st.selectbox("Fasting sugar >120 (fbs)", [0, 1], index=0)

c7, c8 = st.sidebar.columns(2)
with c7:
    restecg = st.number_input("Resting ECG", 0, 2, 0, step=1)
with c8:
    thalach = st.number_input("Max HR (thalach)", 50, 250, 150, step=1)

c9, c10 = st.sidebar.columns(2)
with c9:
    exang = st.selectbox("Exercise angina (exang)", [0, 1], index=0)
with c10:
    oldpeak = st.number_input("ST depression (oldpeak)", value=1.0, format="%.2f")

c11, c12 = st.sidebar.columns(2)
with c11:
    slope = st.number_input("Slope (0–2)", 0, 2, 1, step=1)
with c12:
    ca = st.number_input("Major vessels (ca)", 0, 4, 0, step=1)

thal = st.sidebar.number_input("Thalassemia (thal)", 0, 3, 2, step=1)

# Predict button inside sidebar
if st.sidebar.button("🔍 Predict"):
    x = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                   thalach, exang, oldpeak, slope, ca, thal]])
    try:
        x_scaled = scaler.transform(x)
        pred = int(model.predict(x_scaled)[0])
        prob = None
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(x_scaled)[0][pred])
        label = "Heart Disease" if pred == 1 else "No Heart Disease"
        st.sidebar.success(f"**Result:** {label}")
        if prob is not None:
            st.sidebar.info(f"Confidence: {prob*100:.1f}%")
    except Exception as e:
        st.sidebar.error(f"Prediction error: {e}")

# ---------- Main ----------
st.title("Heart Disease Predictor (Streamlit)")

if df is not None:
    st.markdown("### Exploratory Visuals (scrollable)")
    st.markdown('<div class
