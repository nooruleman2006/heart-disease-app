import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Prediction System",
                   page_icon="❤️", layout="wide")

# ---------- Load model ----------
MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

THAL_MAP = {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}

# ---------- Session defaults and example patients ----------
st.session_state.setdefault("thalach", 150)
st.session_state.setdefault("oldpeak", 1.0)
st.session_state.setdefault("ca", 0)
st.session_state.setdefault("thal_label", "Normal")

def set_example(thalach, oldpeak, ca, thal_label):
    st.session_state.thalach = thalach
    st.session_state.oldpeak = oldpeak
    st.session_state.ca = ca
    st.session_state.thal_label = thal_label

# ---------- Styling (works in both light and dark themes) ----------
st.markdown("""
<style>
footer {visibility: hidden;}

.hero {
    position: relative; overflow: hidden;
    background: linear-gradient(120deg, #ff4b6e 0%, #c2185b 45%, #6a1b9a 100%);
    border-radius: 22px; padding: 34px 36px; color: #fff;
    box-shadow: 0 12px 40px rgba(255, 75, 110, 0.35);
    margin-bottom: 22px;
}
.hero::after {
    content: ""; position: absolute; right: -60px; top: -60px;
    width: 260px; height: 260px; border-radius: 50%;
    background: rgba(255,255,255,0.10);
}
.hero h1 { margin: 0; font-size: 2.4rem; font-weight: 800; color: #fff; }
.hero p  { margin: 8px 0 0 0; font-size: 1.05rem; opacity: 0.92; color: #fff; }
.heart { display: inline-block; animation: beat 1.2s infinite; }
@keyframes beat {
    0%, 100% { transform: scale(1); }
    15% { transform: scale(1.25); }
    30% { transform: scale(1); }
    45% { transform: scale(1.18); }
}
.badges { margin-top: 16px; }
.badge {
    display: inline-block; padding: 5px 14px; margin: 0 8px 6px 0;
    background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.35);
    border-radius: 999px; font-size: 0.85rem; color: #fff;
}

.glass {
    background: rgba(128,128,128,0.08);
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 18px; padding: 20px 22px; margin-bottom: 16px;
}
.section-title { font-size: 1.1rem; font-weight: 700; color: #e91e63; margin-bottom: 4px; }

.result { border-radius: 20px; padding: 24px 28px; margin-top: 10px; }
.result.bad  { background: linear-gradient(135deg, rgba(244,67,54,0.22), rgba(183,28,28,0.10));
               border: 1px solid #ef5350; box-shadow: 0 0 30px rgba(239,83,80,0.25); }
.result.good { background: linear-gradient(135deg, rgba(76,175,80,0.22), rgba(27,94,32,0.10));
               border: 1px solid #66bb6a; box-shadow: 0 0 30px rgba(102,187,106,0.25); }
.result h2 { margin: 0; font-size: 1.9rem; }
.result p  { margin: 8px 0 0 0; opacity: 0.9; }

.gauge-wrap { margin-top: 18px; }
.gauge {
    position: relative; height: 16px; border-radius: 999px;
    background: linear-gradient(90deg, #43a047 0%, #fdd835 50%, #e53935 100%);
}
.marker {
    position: absolute; top: -7px; width: 6px; height: 30px;
    background: #fff; border: 1px solid #555; border-radius: 4px;
    box-shadow: 0 0 10px rgba(0,0,0,0.5);
    transform: translateX(-50%);
}
.gauge-labels { display: flex; justify-content: space-between;
                font-size: 0.78rem; opacity: 0.75; margin-top: 8px; }

.bar-row { margin: 10px 0; }
.bar-head { display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 4px; }
.bar-bg { background: rgba(128,128,128,0.25); border-radius: 999px; height: 9px; }
.bar-fill { height: 9px; border-radius: 999px;
            background: linear-gradient(90deg, #ff4b6e, #ff9a6e); }

.attr-card { border-left: 5px solid #ff4b6e; }
.attr-card h4 { margin: 0 0 4px 0; color: #e91e63; }
.attr-card p { margin: 0; opacity: 0.88; font-size: 0.95rem; }

div.stButton > button {
    border-radius: 14px; font-weight: 700; height: 3rem;
    border: 1px solid rgba(128,128,128,0.35);
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #ff4b6e, #c2185b);
    border: none; font-size: 1.1rem; color: #fff;
    box-shadow: 0 6px 20px rgba(255,75,110,0.4);
}
.small-note { font-size: 0.8rem; opacity: 0.65; text-align: center; margin-top: 18px; }
</style>
""", unsafe_allow_html=True)

# ---------- Hero ----------
st.markdown("""
<div class="hero">
<h1><span class="heart">❤️</span> Heart Disease Prediction System</h1>
<p>An SVM model that estimates heart disease from just four clinical measurements.</p>
<div class="badges">
<span class="badge">UCI Cleveland dataset</span>
<span class="badge">297 patients</span>
<span class="badge">4 key attributes</span>
<span class="badge">Support Vector Classifier</span>
</div>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_attr, tab_model = st.tabs(["🔍 Predict", "🧬 The 4 attributes", "📊 Model performance"])

# ================= TAB 1: PREDICT =================
with tab_predict:
    st.markdown('<div class="section-title">⚡ Try an example patient</div>', unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    e1.button("🔴 Example: higher-risk profile", use_container_width=True,
              on_click=set_example, args=(108, 1.5, 3, "Normal"))
    e2.button("🟢 Example: lower-risk profile", use_container_width=True,
              on_click=set_example, args=(187, 3.5, 0, "Normal"))
    e3.button("↺ Reset", use_container_width=True,
              on_click=set_example, args=(150, 1.0, 0, "Normal"))

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="glass"><div class="section-title">🏃 Exercise test</div>', unsafe_allow_html=True)
        thalach = st.slider("Maximum heart rate achieved (thalach)", 70, 205, key="thalach",
                            help="Highest heart rate reached during the exercise test (beats per minute).")
        oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.5, step=0.1, key="oldpeak",
                            help="How far the ECG ST segment drops during exercise compared with rest.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass"><div class="section-title">🔬 Diagnostic tests</div>', unsafe_allow_html=True)
        ca = st.selectbox("Major vessels colored by fluoroscopy (ca)", [0, 1, 2, 3], key="ca",
                          help="Number of major blood vessels (0 to 3) showing blockage.")
        thal_label = st.selectbox("Thalassemia test result (thal)", list(THAL_MAP.keys()), key="thal_label",
                                  help="Result of the thallium stress test.")
        st.markdown('</div>', unsafe_allow_html=True)

        predict = st.button("🔍 Predict now", type="primary", use_container_width=True)

    thal = THAL_MAP[thal_label]

    with right:
        def bar(label, value_text, frac):
            pct = int(max(0, min(1, frac)) * 100)
            return (f'<div class="bar-row"><div class="bar-head"><span>{label}</span>'
                    f'<b>{value_text}</b></div><div class="bar-bg">'
                    f'<div class="bar-fill" style="width:{pct}%"></div></div></div>')

        bars = (bar("thalach", f"{thalach} bpm", (thalach - 70) / (205 - 70)) +
                bar("oldpeak", f"{oldpeak:.1f}", oldpeak / 6.5) +
                bar("ca", f"{ca} vessel(s)", ca / 3) +
                bar("thal", thal_label, (thal - 3) / 4))
        st.markdown(f'<div class="glass"><div class="section-title">📋 Current patient</div>'
                    f'{bars}<div style="font-size:0.78rem;opacity:0.6;">Bars show where each value sits within its allowed range.</div></div>',
                    unsafe_allow_html=True)

        if predict:
            patient = pd.DataFrame({"thalach": [float(thalach)], "oldpeak": [float(oldpeak)],
                                    "ca": [float(ca)], "thal": [float(thal)]})
            result = int(model.predict(patient)[0])
            score = float(model.decision_function(patient)[0])
            pos = (float(np.clip(score, -2, 2)) + 2) / 4 * 100

            if result == 1:
                cls, icon, title = "bad", "⚠️", "Heart Disease Likely"
                text = "The model predicts that this patient has heart disease. A medical professional should review these results."
            else:
                cls, icon, title = "good", "✅", "No Heart Disease Likely"
                text = "The model predicts that this patient does not have heart disease."

            st.markdown(f"""
<div class="result {cls}">
<h2>{icon} {title}</h2>
<p>{text}</p>
<div class="gauge-wrap">
<div class="gauge"><div class="marker" style="left:{pos:.1f}%"></div></div>
<div class="gauge-labels"><span>Lower-risk side</span><span>Decision boundary</span><span>Higher-risk side</span></div>
</div>
<p style="font-size:0.8rem;opacity:0.7;margin-top:12px;">
Model score: {score:+.2f} (distance from the decision boundary; not a probability).
The closer the marker is to the centre, the more borderline the case.</p>
</div>""", unsafe_allow_html=True)
        else:
            st.info("Set the values, or pick an example, then click **Predict now**.")

# ================= TAB 2: ATTRIBUTES =================
with tab_attr:
    st.markdown("These four attributes had the highest ANOVA F-scores out of the 13 in the dataset.")
    attrs = [
        ("thal", "113.2", "Thalassemia stress test result: 3 = normal, 6 = fixed defect, 7 = reversible defect. Patients with defects have heart disease far more often."),
        ("ca", "80.6", "Number of major vessels (0 to 3) showing blockage on fluoroscopy. Healthy patients mostly have 0."),
        ("oldpeak", "64.7", "ST depression on the ECG during exercise. Higher values mean the heart is under more strain."),
        ("thalach", "64.6", "Maximum heart rate achieved. Patients with disease usually reach a lower maximum."),
    ]
    c1, c2 = st.columns(2)
    for i, (name, f, desc) in enumerate(attrs):
        with (c1 if i % 2 == 0 else c2):
            st.markdown(f'<div class="glass attr-card"><h4>{name} &nbsp;·&nbsp; F-score {f}</h4><p>{desc}</p></div>',
                        unsafe_allow_html=True)

# ================= TAB 3: MODEL =================
with tab_model:
    st.warning("These results come from the same 297 patients the model was trained on, "
               "so accuracy on new patients is likely lower.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", "84%")
    m2.metric("Precision (disease)", "87%")
    m3.metric("Recall (disease)", "76%")
    m4.metric("F1-score (disease)", "81%")

    st.markdown("#### Confusion matrix")
    cm = pd.DataFrame([[145, 15], [33, 104]],
                      index=["Actual: No disease", "Actual: Disease"],
                      columns=["Predicted: No disease", "Predicted: Disease"])
    st.dataframe(cm, use_container_width=True)
    st.caption("33 patients with heart disease were predicted as healthy. This is the model's main weakness.")

st.markdown('<div class="small-note">Learning project by Noor Ul Eman · Not a medical diagnosis</div>',
            unsafe_allow_html=True)
