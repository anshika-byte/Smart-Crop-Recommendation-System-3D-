# =========================
# IMPORT
# =========================
import streamlit as st
import numpy as np
import pickle

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="🌾 Smart Crop ", layout="wide")

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("model2.pkl", "rb"))
scaler = pickle.load(open("scaler2.pkl", "rb"))
crop_labels = pickle.load(open("labels2.pkl", "rb"))

# =========================
# CSS (CLEAN UI ONLY)
# =========================
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1500382017468-9049fed747ef");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Dark overlay */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.45);
    z-index: 0;
}

/* Main container center + spacing */
.block-container {
    position: relative;
    z-index: 1;
    max-width: 1000px;
    padding-left: 80px;
    padding-right: 80px;
    margin: auto;
    text-align: center;
}

/* HEADINGS */
h1 {
    color: white !important;
    font-size: 42px !important;
    text-align: center;
}

h2, h3, h4, h5, h6, p, label {
    color: white !important;
    font-weight: bold !important;
    font-size: 20px !important;
    text-align: center;
}

/* Slider text size */
.stSlider label {
    font-size: 18px !important;
}

/* Selectbox text */
.stSelectbox label {
    font-size: 18px !important;
}

/* BUTTON STYLE */
div.stButton > button {
    background-color: black !important;
    color: black !important;
    font-size: 20px !important;
    font-weight: bold !important;
    padding: 10px 25px !important;
    border-radius: 10px !important;
    border: none !important;
    cursor: pointer !important;
    box-shadow: none !important;
}

/* RESULT BOX (NO GLASS - SIMPLE RED) */
.result-box {
    background: #d50000;
    color: white;
    padding: 20px;
    border-radius: 10px;
    font-size: 28px;
    font-weight: bold;
    margin-top: 20px;
    text-align: center;
}

/* TOP 3 BOX (SIMPLE DARK) */
.top3 {
    background: rgba(0,0,0,0.8);
    color: white;
    padding: 15px;
    border-radius: 10px;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown("<h1>🌾 Smart Crop Recommendation System</h1>", unsafe_allow_html=True)

# =========================
# INPUT UI
# =========================
st.markdown("## 🌱 Enter Soil & Weather Data")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🌿 Soil Nutrients")
    N = st.slider("Nitrogen (N)", 0, 150, 50)
    P = st.slider("Phosphorus (P)", 0, 150, 50)
    K = st.slider("Potassium (K)", 0, 150, 50)

with col2:
    st.markdown("### 🌦 Weather Conditions")
    temp = st.slider("Temperature (°C)", 10, 45, 25)
    humidity = st.slider("Humidity (%)", 10, 100, 60)
    rainfall = st.slider("Rainfall (mm)", 0, 300, 100)

st.markdown("### 🪨 Soil Type")
soil = st.selectbox("", ['sandy', 'loamy', 'clay'])

soil_map = {'sandy': 0, 'loamy': 1, 'clay': 2}

# =========================
# PREDICTION
# =========================

if st.button("🔍 Predict Crop"):

    input_data = np.array([[N, P, K, temp, humidity, rainfall, soil_map[soil]]])
    input_scaled = scaler.transform(input_data)

    probs = model.predict_proba(input_scaled)[0]
    top3 = np.argsort(probs)[-3:][::-1]

    best = crop_labels[top3[0]]

    # RESULT
    st.markdown(
        f"<div class='result-box'>🌾 Recommended Crop: {str(best).upper()}</div>",
        unsafe_allow_html=True
    )

    # TOP 3
    
    st.write("### 🌿 Top 3 Recommendations")

    for i in top3:
        st.write(f"✔ {crop_labels[i]} → {probs[i]*100:.2f}%")

    st.markdown("</div>", unsafe_allow_html=True)
