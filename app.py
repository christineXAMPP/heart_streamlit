import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("heart_disease_model.pkl", "rb"))

# App title
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("❤️ Heart Disease Predictor (Streamlit)")

# Sidebar for patient input
st.sidebar.header("Patient Input")

def user_input_features():
    age = st.sidebar.number_input("Age", 18, 100, 50)
    sex = st.sidebar.selectbox("Sex (0=female, 1=male)", [0, 1])
    cp = st.sidebar.number_input("Chest pain type (cp)", 0, 3, 0)
    trestbps = st.sidebar.number_input("Resting BP (trestbps)", 80, 200, 130)
    chol = st.sidebar.number_input("Cholesterol (chol)", 100, 600, 250)
    fbs = st.sidebar.selectbox("Fasting blood sugar >120 mg/dl (fbs)", [0, 1])
    restecg = st.sidebar.number_input("Resting ECG (restecg)", 0, 2, 0)
    thalach = st.sidebar.number_input("Max heart rate (thalach)", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise induced angina (exang)", [0, 1])
    oldpeak = st.sidebar.number_input("ST depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
    slope = st.sidebar.number_input("Slope (0–2)", 0, 2, 1)
    ca = st.sidebar.number_input("Major vessels (ca)", 0, 4, 0)
    thal = st.sidebar.number_input("Thalassemia (thal)", 0, 3, 2)

    data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])
    return data

# Collect input
input_data = user_input_features()

# Prediction button
if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][prediction][0] * 100

    # Right-side result container
    with st.container():
        st.subheader("🩺 Prediction Result")
        if prediction[0] == 1:
            st.error("⚠️ Heart Disease Detected")
        else:
            st.success("✅ No Heart Disease")

        st.info(f"Confidence: {probability:.1f}%")

        st.caption("Feature order: age, sex, cp, trestbps, chol, fbs, restecg, "
                   "thalach, exang, oldpeak, slope, ca, thal")
else:
    st.write("⬅️ Enter details in the sidebar and click **Predict** to see results here.")
