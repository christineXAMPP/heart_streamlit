# app.py
import streamlit as st
import numpy as np
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
    return model, scaler

model, scaler = load_artifacts()

# ---------- UI ----------
st.title("❤️ Heart Disease Predictor")

# Sidebar inputs
st.sidebar.header("Patient Input")

col1, col2 = st.sidebar.columns(2)
age = col1.number_input("Age", min_value=1, max_value=120, value=50)
sex = col2.selectbox("Sex (0=female, 1=male)", options=[0, 1], index=1)

col3, col4 = st.sidebar.columns(2)
cp = col3.number_input("Chest pain type (cp)", min_value=0, max_value=3, value=0)
trestbps = col4.number_input("Resting BP", min_value=50, max_value=250, value=130)

col5, col6 = st.sidebar.columns(2)
chol = col5.number_input("Cholesterol", min_value=100, max_value=600, value=250)
fbs = col6.selectbox("Fasting sugar >120 (fbs)", options=[0, 1], index=0)

col7, col8 = st.sidebar.columns(2)
restecg = col7.number_input("Resting ECG", min_value=0, max_value=2, value=0)
thalach = col8.number_input("Max heart rate", min_value=50, max_value=250, value=150)

col9, col10 = st.sidebar.columns(2)
exang = col9.selectbox("Exercise angina (exang)", options=[0, 1], index=0)
oldpeak = col10.number_input("ST depression", value=1.0, format="%.2f")

col11, col12 = st.sidebar.columns(2)
slope = col11.number_input("Slope (0-2)", min_value=0, max_value=2, value=1)
ca = col12.number_input("Major vessels (0-4)", min_value=0, max_value=4, value=0)

thal = st.sidebar.number_input("Thalassemia (0-3)", min_value=0, max_value=3, value=2)

predict_btn = st.sidebar.button("🔍 Predict", use_container_width=True)

# ---------- Main Result Section ----------
if predict_btn:
    x = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                   thalach, exang, oldpeak, slope, ca, thal]])
    try:
        x_scaled = scaler.transform(x)
    except Exception as e:
        st.error(f"Error scaling input: {e}")
    else:
        pred = int(model.predict(x_scaled)[0])
        prob = None
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(x_scaled)[0][pred])

        label = "Heart Disease" if pred == 1 else "No Heart Disease"

        with st.container():
            st.subheader("🩺 Prediction Result")
            st.markdown(
                f"""
                <div style="padding:15px; border-radius:10px; background-color:#f9f9f9; 
                            border:1px solid #ddd; text-align:center;">
                    <h3 style="color:#333;">Result: <span style="color:{'red' if pred==1 else 'green'};">{label}</span></h3>
                    {'<p style="font-size:18px;">Confidence: {:.1f}%</p>'.format(prob*100) if prob else ''}
                </div>
                """,
                unsafe_allow_html=True
            )

st.markdown("---")
st.caption("Model feature order: age, sex, cp, trestbps, chol, fbs, restecg, "
           "thalach, exang, oldpeak, slope, ca, thal")
