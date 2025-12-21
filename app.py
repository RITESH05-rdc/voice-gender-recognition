import streamlit as st
import librosa
import librosa.display
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Voice Gender Recognition",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.card {
    background-color: rgba(255, 255, 255, 0.05);
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.1);
}
.center {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="center">
    <h1>🎤 Voice Gender Recognition</h1>
    <p>Upload a WAV audio file to predict gender</p>
</div>
""", unsafe_allow_html=True)

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050, duration=3)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0), mfcc, sr, audio

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("audio_gender_model.pkl")

model = load_model()

# ---------------- UPLOAD CARD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📁 Upload audio file (.wav)", type=["wav"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    features, mfcc, sr, audio = extract_features("temp.wav")
    features = features.reshape(1, -1)

    # ---------------- AUDIO CARD ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎧 Audio Preview")
    st.audio(uploaded_file, format="audio/wav")
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- MFCC CARD ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 MFCC Visualization")

    fig, ax = plt.subplots(figsize=(7, 3))
    img = librosa.display.specshow(
        mfcc,
        x_axis="time",
        cmap="magma",
        ax=ax
    )
    ax.set(title="MFCC Heatmap")
    fig.colorbar(img, ax=ax, format="%+2.f")
    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- PREDICT BUTTON ----------------
    st.markdown('<div class="center">', unsafe_allow_html=True)
    predict = st.button("🔍 Predict Gender")
    st.markdown('</div>', unsafe_allow_html=True)

    if predict:
        prediction = model.predict(features)
        gender = "Male 👨" if prediction[0] == 1 else "Female 👩"

        st.markdown('<div class="card center">', unsafe_allow_html=True)
        st.success(f"Predicted Gender: **{gender}**")
        st.markdown('</div>', unsafe_allow_html=True)
