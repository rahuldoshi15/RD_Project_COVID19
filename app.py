import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("covid_xray_model.h5")

# Class labels
class_names = ["COVID", "Normal", "Viral Pneumonia"]

st.title("COVID-19 Detection from Chest X-ray")
st.write("Upload a chest X-ray image to classify")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess image
    img = np.array(image)
    img = cv2.resize(img, (128, 128))   # MUST match training size
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display results
    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    st.bar_chart(prediction[0])
