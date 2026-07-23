import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load model
model = tf.keras.models.load_model("best_model.h5")

# Load class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿")
st.title("🌿 Plant Leaf Disease Detection System")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image",  width=700)

    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    confidence = np.max(prediction)
    class_index = np.argmax(prediction)

    label = class_names[class_index]
    plant, disease = label.split("___")

    st.success(f"🌱 Plant : {plant}")
    st.warning(f"🦠 Disease : {disease}")
    st.info(f"📊 Confidence : {confidence*100:.2f}%")
