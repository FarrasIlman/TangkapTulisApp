import streamlit as st
st.set_page_config(page_title="Handwriting App", page_icon="✍️")  # Harus di baris 2

import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

@st.cache_resource
def load_model_file():
    return load_model("model50v2.keras")

model = load_model_file()

alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "

def num_to_label(num):
    result = ""
    for ch in num:
        if ch == -1:
            break
        result += alphabets[ch]
    return result

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
st.title("📝 Handwriting Recognition")
uploaded_file = st.file_uploader("Upload a handwriting image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    decoded = K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1], greedy=True)[0][0]
    decoded_text = num_to_label(tf.keras.backend.get_value(decoded)[0])

    st.success(f"Predicted Text: {decoded_text}")
