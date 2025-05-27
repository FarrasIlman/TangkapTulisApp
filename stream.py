import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from keras.models import load_model
from keras import backend as K

# Load model
@st.cache_resource
def load_recognition_model():
    model = load_model('model/model.keras')
    return model

model = load_recognition_model()

# Alfabet dan fungsi konversi
alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
max_str_len = 24
num_of_characters = len(alphabets) + 1
num_of_timestamps = 64

def label_to_num(label):
    return [alphabets.find(ch) for ch in label]

def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:
            break
        ret += alphabets[ch]
    return ret

# Preprocessing gambar
def preprocess(img):
    img = np.array(img.convert('L'))  # Convert to grayscale
    (h, w) = img.shape
    final_img = np.ones([64, 256]) * 255  # White background

    if w > 256:
        img = img[:, :256]
    if h > 64:
        img = img[:64, :]

    final_img[:h, :w] = img
    final_img = cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)
    final_img = final_img / 255.0
    return final_img.reshape(1, 256, 64, 1)

# Streamlit UI
st.set_page_config(page_title="Handwriting Recognition", layout="centered")
st.title("📝 Handwriting Recognition")
st.write("Upload gambar tulisan tangan dan model akan memprediksi teksnya.")

uploaded_file = st.file_uploader("📤 Upload gambar tulisan tangan (JPEG/PNG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption='📷 Gambar yang Diupload', use_column_width=True)

    with st.spinner("🔍 Memproses gambar..."):
        processed_image = preprocess(image)
        pred = model.predict(processed_image)
        decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1],
                                           greedy=True)[0][0])
        result = num_to_label(decoded[0])

    st.success("✅ Prediksi Selesai!")
    st.markdown(f"""
    ### ✨ Hasil Prediksi:
    <h2 style='text-align: center; color: green;'>{result}</h2>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Model handwriting recognition berbasis CNN+BiLSTM+CTC loss")
else:
    st.info("Silakan upload gambar tulisan tangan terlebih dahulu.")
