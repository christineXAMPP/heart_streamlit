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
    except Exception as e:
        st.error("Model file not found or failed to load. Make sure model_knn.joblib is in the app folder.")
        st.stop()
    try:
        scaler = load(base / "scaler.joblib")
    except Exception as e:
        st.error("Scaler file not found or failed to load. Make sure scaler.joblib is in the app folder.")
        st.stop()
    try:
        df = pd.read_csv(base / "heart.csv")
    except Exception:
        df = None
    return model, scaler, df

model, scaler, df = load_artifacts()

# ---------- Styles (make button bigger, widen sidebar inputs) ----------
st.markdown(
    """
    <style>
    /* Make the primary button full width and larger */
    div.stButton > button {
        width: 100%;
        height: 48px;
        font-size: 16px;
        border-radius: 8px;
    }

    /* Slightly widen the sidebar content area */
    .css-1d391kg { /* streamlit class may change; this helps in many versions */
        max-width: 380px;
    }

    /* Increase spacing for main heading */
    h1 {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar: 2-column layout for inputs ----------
st.sidebar.header("Patient Input")

# Use two columns per row inside the sidebar
c1, c2 = st.sidebar.columns(2)
with c1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1)
with c2:
    sex = st.selectbox("Sex (0=female, 1=male)", options=[0, 1], index=1)

c3, c4 = st.sidebar.columns(2)
with c3:
    cp = st.number_input("Chest pain type (cp)", min_value=0, max_value=3, value=0, step=1)
with c4:
    trestbps = st.number_input("Resting BP (trestbps)", min_value=50, max_value=250, value=130, step=1)

c5, c6 = st.sidebar.columns(2)
with c5:
    chol = st.number_input("Cholesterol (chol)", min_value=100, max_value=600, value=250, step=1)
with c6:
    fbs = st.selectbox("Fasting blood sugar >120 mg/dl (fbs)", options=[0, 1], index=0)

c7, c8 = st.sidebar.columns(2)
with c7:
    restecg = st.number_input("Resting ECG (restecg)", min_value=0, max_value=2, value=0, step=1)
with c8:
    thalach = st.number_input("Max heart rate (thalach)", min_value=50, max_value=250, value=150, step=1)

c9, c10 = st.sidebar.columns(2)
with c9:
    exang = st.selectbox("Exercise induced angina (exang)", options=[0, 1], index=0)
with c10:
    oldpeak = st.number_input("ST depression (oldpeak)", value=1.0, format="%.2f")

c11, c12 = st.sidebar.columns(2)
with c11:
    slope = st.number_input("Slope (0,1,2)", min_value=0, max_value=2, value=1, step=1)
with c12:
    ca = st.number_input("Major vessels (ca)", min_value=0, max_value=4, value=0, step=1)

# Full width last item if needed
thal = st.sidebar.number_input("Thalassemia (thal)", min_value=0, max_value=3, value=2, step=1)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: provide sensible ranges for reliable predictions")

# ---------- Main layout ----------
st.title("Heart Disease Predictor (Streamlit)")

# Put the Predict button in a small centered column group in the main area
btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
with btn_col2:
    if st.button("Predict"):
        # build input array in expected order
        x = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        try:
            x_scaled = scaler.transform(x)
            pred = int(model.predict(x_scaled)[0])
            prob = None
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(x_scaled)[0][pred])
            label = "Heart Disease" if pred == 1 else "No Heart Disease"
            st.subheader("Prediction Result")
            st.write(f"**Label:** {label}")
            if prob is not None:
                st.write(f"**Confidence:** {prob*100:.1f}%")
            st.caption("Model feature order: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ---------- Visuals (two-column main content) ----------
if df is not None:
    st.markdown("## Exploratory Visuals (dataset)")
    left, right = st.columns([2, 3])

    with left:
        st.write("Distribution of heart disease cases")
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        sns.countplot(x='target', data=df, palette='coolwarm', ax=ax1)
        ax1.set_xlabel("Heart Disease (1 = Yes, 0 = No)")
        st.pyplot(fig1)

        st.write("Cholesterol by Heart Disease (boxplot)")
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        sns.boxplot(x='target', y='chol', data=df, palette='Set2', ax=ax3)
        st.pyplot(fig3)

    with right:
        st.write("Age vs Cholesterol")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.scatterplot(x='age', y='chol', hue='target', data=df, palette='coolwarm', ax=ax2)
        st.pyplot(fig2)

        st.write("Resting blood pressure distribution")
        fig4, ax4 = plt.subplots(figsize=(6, 3))
        sns.histplot(df['trestbps'], bins=30, kde=True, ax=ax4)
        st.pyplot(fig4)
else:
    st.info("Dataset `heart.csv` not found in app folder. Add it to enable visuals.")

st.markdown("---")
st.write("App by you — improved UI: two-column sidebar and larger predict button.")
