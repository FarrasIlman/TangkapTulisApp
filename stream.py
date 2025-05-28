import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Konfigurasi halaman
st.set_page_config(page_title="Tangkap Tulis ✍️", page_icon="📝", layout="centered")

# Gaya CSS kustom
st.markdown("""
    <style>
    .title {
        font-size: 45px;
        font-weight: bold;
        text-align: center;
        color: #2E8B57;
        margin-bottom: 0px;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #555;
        margin-top: 5px;
    }
    .upload-section {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
    }
    .prediction-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #eee;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Load model (cache)
@st.cache_resource
def load_model_file():
    return load_model("model50v2.keras")

model = load_model_file()

# Karakter valid
alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "

# Fungsi decoding CTC
def ctc_decode(prediction):
    pred_indices = np.argmax(prediction, axis=2)[0]
    previous_char = -1
    decoded = ""

    for ch in pred_indices:
        ch = int(ch)
        if ch == previous_char or ch == -1:
            continue
        if 0 <= ch < len(alphabets):
            decoded += alphabets[ch]
        previous_char = ch
    return decoded

# Preprocessing gambar
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape
    final_img = np.ones([64, 256]) * 255
    if w > 256:
        image = image[:, :256]
    if h > 64:
        image = image[:64, :]
    final_img[:h, :w] = image
    final_img = cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)
    final_img = final_img / 255.0
    final_img = final_img.reshape(1, 256, 64, 1)
    return final_img

# Judul
st.markdown('<div class="title">📝 Tangkap Tulis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Tangkap tulisan tanganmu dan ubah menjadi teks secara otomatis!</div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Upload gambar
with st.container():
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Unggah Gambar Tulisan (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)

# Prediksi jika file diunggah
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, caption="🖼️ Gambar Terunggah", use_container_width=True)

    # Prediksi
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_label = ctc_decode(prediction)

    # Tampilkan hasil prediksi
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📄 Hasil Prediksi")
    st.markdown('<div class="prediction-box"><h3 style="color:#2E8B57;">🧠 Teks Terbaca:</h3><p style="font-size:22px;">' + predicted_label + '</p></div>', unsafe_allow_html=True)
