import streamlit as st
import numpy as np
from PIL import Image
from google import genai as google_genai
import time
import gdown
import os

# ======================
# CONFIG
# ======================
MODEL_PATH = "best_model4.h5"
IMG_SIZE = 224
GEMINI_API_KEY = "AIzaSyBB3pSmoqBsSOauVSU4Zq79tPWCVvu_GsQ"
FILE_ID='1IgfEyB16Fx8w17DonPi2J4S-loBrjEXz'
# ✅ FIX: Use new google-genai SDK with current model (gemini-1.5-flash is shut down)
# Run: pip install google-genai
gemini_client = google_genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash"  # stable, widely available model

# ✅ FIX: Lazy-load TensorFlow to avoid crashing if not installed
@st.cache_resource
def load_model():
    try:
        import tensorflow as tf
        if not os.path.exists(MODEL_PATH):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        return None

class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

tumor_info = {
    "glioma":      {"icon": "🔴", "color": "#FF4B6E", "badge": "HIGH RISK"},
    "meningioma":  {"icon": "🟠", "color": "#FF9A3C", "badge": "MODERATE"},
    "no_tumor":    {"icon": "🟢", "color": "#00D4AA", "badge": "CLEAR"},
    "pituitary":   {"icon": "🟡", "color": "#FFD166", "badge": "MODERATE"},
}

# ======================
# PAGE CONFIG & CSS
# ======================
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@300;400&display=swap');

/* ---- ROOT VARIABLES ---- */
:root {
    --bg:       #050A0F;
    --surface:  #0D1B2A;
    --card:     #0F2236;
    --border:   #1A3A52;
    --text:     #E2EEF9;
    --muted:    #5A7A96;
    --accent:   #00D4FF;
    --accent2:  #00FFB3;
    --danger:   #FF4B6E;
    --warn:     #FF9A3C;
}

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: var(--bg);
    color: var(--text);
}

/* ---- HIDE STREAMLIT CHROME ---- */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1200px; }

/* ---- ANIMATED GRID BG ---- */
body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ---- HEADER ---- */
.neuro-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 2rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}
.neuro-logo {
    width: 54px; height: 54px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px;
    box-shadow: 0 0 30px rgba(0,212,255,0.3);
}
.neuro-title { font-family: 'Syne', sans-serif; }
.neuro-title h1 {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1;
}
.neuro-title p {
    margin: 4px 0 0;
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* ---- STATUS BADGE ---- */
.status-badge {
    margin-left: auto;
    padding: 6px 14px;
    border: 1px solid var(--accent2);
    border-radius: 20px;
    font-size: 0.68rem;
    color: var(--accent2);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.5; }
}

/* ---- UPLOAD ZONE ---- */
.upload-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
[data-testid="stFileUploader"] > div {
    background: var(--card) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 16px !important;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: var(--accent) !important;
}

/* ---- SECTION CARD ---- */
.scan-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
}
.scan-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.scan-card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
}

/* ---- RESULT BOX ---- */
.result-main {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.2rem 1.4rem;
    border-radius: 14px;
    margin-bottom: 1rem;
    border: 1px solid var(--border);
}
.result-icon { font-size: 2.2rem; line-height: 1; }
.result-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
}
.result-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    line-height: 1.1;
    margin: 2px 0;
}
.result-badge {
    margin-left: auto;
    padding: 5px 12px;
    border-radius: 6px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ---- CONFIDENCE BAR ---- */
.conf-row { margin-bottom: 0.6rem; }
.conf-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: var(--muted);
    margin-bottom: 4px;
}
.conf-track {
    height: 6px;
    background: var(--surface);
    border-radius: 3px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 1s ease;
}

/* ---- AI ADVICE CARD ---- */
.advice-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-top: 1rem;
    line-height: 1.8;
    font-size: 0.88rem;
    color: var(--text);
}
.advice-card h3 {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--accent);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 0.8rem;
}

/* ---- METRICS ROW ---- */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.8rem;
    margin-bottom: 1.2rem;
}
.metric-tile {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--accent);
}
.metric-lbl {
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-top: 2px;
}

/* ---- PREDICT BUTTON ---- */
.stButton > button {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
    color: #050A0F !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(0,212,255,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 30px rgba(0,212,255,0.4) !important;
}

/* ---- DISCLAIMER ---- */
.disclaimer {
    margin-top: 2.5rem;
    padding: 1rem 1.4rem;
    background: rgba(255,75,110,0.06);
    border: 1px solid rgba(255,75,110,0.2);
    border-radius: 12px;
    font-size: 0.72rem;
    color: var(--muted);
    line-height: 1.6;
}
.disclaimer span { color: var(--danger); font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ======================
# HEADER
# ======================
st.markdown("""
<div class="neuro-header">
    <div class="neuro-logo">🧠</div>
    <div class="neuro-title">
        <h1>NeuroScan AI</h1>
        <p>Brain Tumor Detection System · v2.0</p>
    </div>
    <div class="status-badge">● System Online</div>
</div>
""", unsafe_allow_html=True)

# ======================
# LAYOUT
# ======================
left, right = st.columns([1, 1.1], gap="large")

with left:
    st.markdown('<div class="scan-card">', unsafe_allow_html=True)
    st.markdown('<div class="scan-card-title">📂 MRI Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="", use_container_width=True)  # ✅ FIX: use_column_width → use_container_width

        st.markdown('<br>', unsafe_allow_html=True)
        predict_clicked = st.button("⚡ Run Analysis")
    else:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem; color: #3A6080;">
            <div style="font-size:3rem">🩻</div>
            <div style="font-size:0.78rem; margin-top:0.5rem; letter-spacing:0.1em; text-transform:uppercase;">
                Upload an MRI scan to begin
            </div>
        </div>
        """, unsafe_allow_html=True)
        predict_clicked = False

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    if uploaded_file and predict_clicked:
        model = load_model()

        if model is not None:
            # Preprocess
            img_arr = np.array(image.resize((IMG_SIZE, IMG_SIZE))) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)

            with st.spinner("Analyzing scan..."):
                prediction = model.predict(img_arr)
                time.sleep(0.3)  # slight delay for UX feel

            predicted_class = class_names[np.argmax(prediction)]
            confidence = float(np.max(prediction)) * 100
            info = tumor_info[predicted_class]

            # ---- RESULT CARD ----
            st.markdown('<div class="scan-card">', unsafe_allow_html=True)
            st.markdown('<div class="scan-card-title">🧾 Diagnosis Result</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="result-main" style="background: {info['color']}12; border-color: {info['color']}40;">
                <div class="result-icon">{info['icon']}</div>
                <div>
                    <div class="result-label">Detected Condition</div>
                    <div class="result-name" style="color:{info['color']}">
                        {predicted_class.replace('_',' ').title()}
                    </div>
                </div>
                <div class="result-badge" style="background:{info['color']}22; color:{info['color']}; border:1px solid {info['color']}55;">
                    {info['badge']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ---- METRICS ----
            second_idx = np.argsort(prediction[0])[-2]
            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-tile">
                    <div class="metric-val">{confidence:.1f}%</div>
                    <div class="metric-lbl">Confidence</div>
                </div>
                <div class="metric-tile">
                    <div class="metric-val">{class_names[second_idx].replace('_',' ').title()[:8]}</div>
                    <div class="metric-lbl">2nd Match</div>
                </div>
                <div class="metric-tile">
                    <div class="metric-val">{len(class_names)}</div>
                    <div class="metric-lbl">Classes</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ---- CLASS BARS ----
            st.markdown('<div class="scan-card-title" style="margin-top:0.5rem">📊 Class Probabilities</div>', unsafe_allow_html=True)
            bar_colors = ["#FF4B6E","#FF9A3C","#00D4AA","#FFD166"]
            for i, (name, prob) in enumerate(zip(class_names, prediction[0])):
                pct = float(prob) * 100
                st.markdown(f"""
                <div class="conf-row">
                    <div class="conf-header">
                        <span>{name.replace('_',' ').title()}</span>
                        <span>{pct:.1f}%</span>
                    </div>
                    <div class="conf-track">
                        <div class="conf-fill" style="width:{pct}%; background:{bar_colors[i]};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # ---- AI ADVICE ----
            st.markdown('<div class="scan-card-title" style="margin-top:1.2rem">🤖 AI Care Suggestions</div>', unsafe_allow_html=True)

            with st.spinner("Generating AI advice..."):
                try:
                    prompt = f"""
A patient has been flagged for {predicted_class.replace('_', ' ')} in a brain MRI scan.
Provide:
1. Simple explanation of the condition
2. Key precautions
3. Lifestyle advice
4. When to urgently consult a doctor

Be concise, compassionate, and easy to understand. Use short paragraphs.
"""
                    response = gemini_client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=prompt
                    )
                    advice_text = response.text
                except Exception as e:
                    advice_text = f"⚠️ AI service temporarily unavailable. Please consult a medical professional directly.\n\nError: {e}"

            st.markdown(f'<div class="advice-card">{advice_text}</div>', unsafe_allow_html=True)

    elif not uploaded_file:
        st.markdown("""
        <div class="scan-card" style="height:100%; min-height: 300px; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:1rem; color:#2A5070; padding: 4rem 2rem; text-align:center;">
            <div style="font-size:3.5rem">🔬</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; color:#1A4060;">
                Awaiting Scan Input
            </div>
            <div style="font-size:0.75rem; letter-spacing:0.08em; max-width:240px; line-height:1.8; color:#2A5070;">
                Upload an MRI image on the left panel to run the neural network analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ======================
# DISCLAIMER
# ======================
st.markdown("""
<div class="disclaimer">
    <span>⚠ Medical Disclaimer:</span> This tool is intended for <strong>research and educational purposes only</strong>.
    It does not constitute medical advice, diagnosis, or treatment. Always consult a qualified
    neurologist or radiologist for clinical decisions.
</div>
""", unsafe_allow_html=True)
