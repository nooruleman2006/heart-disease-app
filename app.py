import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Prediction System", page_icon="❤️")

# Load the saved model (scaler + SVC) from the same folder as this file
MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

# ---------- Header ----------
st.title("❤️ Heart Disease Prediction System")
st.write("Enter the patient's values below and click **Predict**.")
st.caption("Model: Support Vector Classifier trained on the Cleveland dataset "
           "using 4 attributes: thal, ca, oldpeak and thalach.")

# ---------- Inputs ----------
col1, col2 = st.columns(2)

with col1:
    thalach = st.slider("Maximum heart rate achieved (thalach)", 70, 205, 150)
    oldpeak = st.slider("ST depression induced by exercise (oldpeak)", 0.0, 6.5, 1.0, 0.1)

with col2:
    ca = st.selectbox("Major vessels colored by fluoroscopy (ca)", [0, 1, 2, 3])
    thal_label = st.selectbox(
        "Thalassemia (thal)",
        ["Normal", "Fixed defect", "Reversible defect"]
    )

thal = {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}[thal_label]

# ---------- Prediction ----------
if st.button("Predict", type="primary"):
    patient = pd.DataFrame({
        "thalach": [float(thalach)],
        "oldpeak": [float(oldpeak)],
        "ca": [float(ca)],
        "thal": [float(thal)],
    })
    result = model.predict(patient)[0]

    if result == 1:
        st.error("Heart Disease Likely: the model predicts that this patient has heart disease.")
    else:
        st.success("No Heart Disease Likely: the model predicts that this patient does not have heart disease.")

    st.write(f"Inputs used: thalach = {thalach}, oldpeak = {oldpeak:.1f}, ca = {ca}, thal = {thal}")

st.divider()
st.caption("This is a learning project, not a medical diagnosis.")