import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# HARUS di atas sebelum perintah Streamlit lainnya
st.set_page_config(page_title="Handwriting Recognition", page_icon="📝")

# Load model
@st.cache_resource
def load_handwriting_model():
    return load_model("model50v2.keras")

model = load_handwriting_model()

alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "

def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:
            break
        else:
            ret += alphabets[ch]
    return ret

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

# Streamlit UI
st.title("📜 Handwriting Recognition (IAM Dataset)")
st.write("Upload gambar tulisan tangan dan model akan mengenali teksnya.")

uploaded_file = st.file_uploader("Upload Gambar", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, caption='Gambar Diunggah', use_column_width=True)

    processed_image = preprocess_image(image)

    pred = model.predict(processed_image)
    decoded = K.ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=True)[0][0]
    decoded_text = num_to_label(tf.keras.backend.get_value(decoded)[0])

    st.markdown("### ✨ Hasil Prediksi:")
    st.success(decoded_text)
