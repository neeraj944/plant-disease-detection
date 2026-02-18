import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image


# CONFIG

MODEL_PATH = "plant_disease_model_finetuned.h5"   #  fine-tuned model
DATASET_DIR = "dataset/PlantVillage"
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 70.0  

st.set_page_config(
    page_title="Plant Disease Detection",
    layout="centered"
)


# LOAD MODEL

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Load class names
class_names = sorted(os.listdir(DATASET_DIR))


# UI

st.title("🌱 Plant Disease Detection Using Deep Learning")
st.write(
    "Upload an affected leaf image and the model will predict "
    "the plant disease using a fine-tuned deep learning model."
)

uploaded_file = st.file_uploader(
    " Upload leaf image",
    type=["jpg", "jpeg", "png"]
)


# PREDICTION

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    # Preprocess image
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.subheader(" Prediction Result")

    # Confidence-based decision
    if confidence < CONFIDENCE_THRESHOLD:
        st.warning(
            f" Low confidence prediction ({confidence:.2f}%). "
            "Please upload a clearer image or consult an expert."
        )
    else:
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        if "healthy" in predicted_class.lower():
            st.success(" The plant appears to be healthy")
        else:
            st.error("⚠️ Disease detected")


# FOOTER

st.markdown("---")
st.caption(
    "Model: MobileNetV2 (Transfer Learning + Fine-Tuning) | "
    "Project: Plant Disease Detection Using Deep Learning"
)
