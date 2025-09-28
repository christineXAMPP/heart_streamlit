# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from pathlib import Path

# ---------- Main layout: 3-column with margins ----------
st.title("Heart Disease Predictor (Streamlit)")

# Add more balanced spacing: left margin, main charts, right container
left_margin, main_col, right_col, right_margin = st.columns([0.2, 3, 1.2, 0.2])

# Right column: wrap in container card
with right_col:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown("<h3>🩺 Prediction Result</h3>", unsafe_allow_html=True)

    if "prediction_error" in st.session_state:
        st.error(f"Prediction error: {st.session_state.pop('prediction_error')}")
    elif "prediction" in st.session_state:
        p = st.session_state["prediction"]
        conf = st.session_state.get("confidence", None)

        if p == "Heart Disease":
            st.markdown(
                f'<div class="result-status"><strong style="color:#ffb4b4">⚠️ {p}</strong></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-status"><strong style="color:#9be7a2">✅ {p}</strong></div>',
                unsafe_allow_html=True,
            )

        if conf is not None:
            st.markdown(
                f'<div class="result-confidence">Confidence: <strong>{conf:.1f}%</strong></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.caption(
            "Feature order: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal"
        )
    else:
        st.info("No prediction yet. Use the sidebar inputs and click Predict.")

    st.markdown('</div>', unsafe_allow_html=True)

# Main column: charts inside scrollable card with spacing
with main_col:
    st.markdown(
        '<div class="scroll-container" style="margin-right:20px;margin-left:20px;">',
        unsafe_allow_html=True,
    )
    st.markdown("## 📊 Exploratory Visuals")

    if df is not None:
        st.write("Distribution of heart disease cases")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.countplot(x="target", data=df, palette="coolwarm", ax=ax1)
        ax1.set_xlabel("Heart Disease (1 = Yes, 0 = No)")
        st.pyplot(fig1, clear_figure=True)

        st.write("Age vs Cholesterol")
        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        sns.scatterplot(x="age", y="chol", hue="target", data=df, palette="coolwarm", ax=ax2)
        st.pyplot(fig2, clear_figure=True)

        st.write("Cholesterol by Heart Disease (boxplot)")
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        sns.boxplot(x="target", y="chol", data=df, palette="Set2", ax=ax3)
        st.pyplot(fig3, clear_figure=True)

        st.write("Resting blood pressure distribution")
        fig4, ax4 = plt.subplots(figsize=(7, 4))
        sns.histplot(df["trestbps"], bins=30, kde=True, ax=ax4)
        st.pyplot(fig4, clear_figure=True)
    else:
        st.info("Dataset `heart.csv` not found — upload it to enable visuals.")

    st.markdown("</div>", unsafe_allow_html=True)

