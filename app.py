import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="centered",
)

# ---------- Load model ----------
MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

# ---------- Custom styling ----------
st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #b71c1c 0%, #e53935 55%, #ff7043 100%);
    padding: 30px 28px;
    border-radius: 16px;
    color: white;
    text-align: center;
    box-shadow: 0 6px 18px rgba(183, 28, 28, 0.35);
    margin-bottom: 22px;
}
.hero h1 { margin: 0; font-size: 2.1rem; color: white; }
.hero p  { margin: 8px 0 0 0; font-size: 1.02rem; opacity: 0.95; }

.section-title {
    font-size: 1.15rem;
    font-weight: 600;
    margin: 18px 0 6px 0;
    color: #e53935;
}

.result-card {
    padding: 22px 26px;
    border-radius: 14px;
    margin-top: 14px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
}
.result-bad  { background: #ffebee; border-left: 9px solid #c62828; }
.result-good { background: #e8f5e9; border-left: 9px solid #2e7d32; }
.result-title { font-size: 1.6rem; font-weight: 700; margin: 0; }
.result-bad  .result-title { color: #b71c1c; }
.result-good .result-title { color: #1b5e20; }
.result-text { font-size: 1rem; margin-top: 6px; color: #333333; }

div.stButton > button {
    width: 100%;
    height: 3rem;
    font-size: 1.1rem;
    font-weight: 600;
    border-radius: 12px;
}
.footer {
    text-align: center;
    color: #888888;
    font-size: 0.85rem;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About this app")
    st.write(
        "A Support Vector Classifier trained on the UCI Cleveland heart "
        "disease dataset (297 patients), using the 4 most discriminating "
        "attributes."
    )
    st.subheader("The 4 attributes")
    st.markdown(
        "- **thal**: thalassemia test result\n"
        "- **ca**: major vessels with blockage\n"
        "- **oldpeak**: ST depression during exercise\n"
        "- **thalach**: maximum heart rate achieved"
    )
    st.subheader("Model performance")
    st.metric("Training accuracy", "84%")
    st.caption("Measured on the same data the model was trained on, so real-world accuracy is likely lower.")

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <h1>❤️ Heart Disease Prediction System</h1>
    <p>Enter the patient's values and let the model estimate the result.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Inputs ----------
st.markdown('<div class="section-title">🩺 Exercise test measurements</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    thalach = st.slider(
        "Maximum heart rate achieved (thalach)", 70, 205, 150,
        help="The highest heart rate reached during the exercise test (beats per minute).")
with c2:
    oldpeak = st.slider(
        "ST depression (oldpeak)", 0.0, 6.5, 1.0, 0.1,
        help="How much the ECG ST segment drops during exercise compared with rest.")

st.markdown('<div class="section-title">🔬 Diagnostic tests</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    ca = st.selectbox(
        "Major vessels colored by fluoroscopy (ca)", [0, 1, 2, 3],
        help="Number of major blood vessels (0 to 3) showing blockage.")
with c4:
    thal_label = st.selectbox(
        "Thalassemia test result (thal)",
        ["Normal", "Fixed defect", "Reversible defect"],
        help="Result of the thallium stress test.")

thal = {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}[thal_label]

st.write("")

# ---------- Prediction ----------
if st.button("🔍 Predict", type="primary"):
    patient = pd.DataFrame({
        "thalach": [float(thalach)],
        "oldpeak": [float(oldpeak)],
        "ca": [float(ca)],
        "thal": [float(thal)],
    })
    result = model.predict(patient)[0]

    if result == 1:
        st.markdown("""
        <div class="result-card result-bad">
            <p class="result-title">⚠️ Heart Disease Likely</p>
            <p class="result-text">The model predicts that this patient has heart disease.
            A medical professional should review these results.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-card result-good">
            <p class="result-title">✅ No Heart Disease Likely</p>
            <p class="result-text">The model predicts that this patient does not have heart disease.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">📋 Values used for this prediction</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("thalach", thalach)
    m2.metric("oldpeak", f"{oldpeak:.1f}")
    m3.metric("ca", ca)
    m4.metric("thal", thal_label)

st.divider()
st.markdown(
    '<div class="footer">Learning project by Noor Ul Eman. '
    'Not a medical diagnosis.</div>',
    unsafe_allow_html=True,
)
