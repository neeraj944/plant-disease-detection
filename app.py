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

def load_class_names(dataset_dir):
    # Keep only class directories so hidden files like .DS_Store do not break mapping.
    return sorted(
        entry
        for entry in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, entry))
    )

def format_prediction_label(raw_label):
    normalized = raw_label.replace("___", "|")

    if "|" in normalized:
        plant, condition = normalized.split("|", 1)
    elif normalized.lower().endswith("_healthy"):
        plant = normalized[: -len("_healthy")]
        condition = "healthy"
    else:
        plant, _, condition = normalized.partition("_")

    plant_name = plant.replace("__", " ").replace("_", " ").strip().title()
    condition_name = condition.replace("__", " ").replace("_", " ").strip().title()

    if not plant_name:
        plant_name = "Unknown"
    if not condition_name:
        condition_name = "Unknown"

    return plant_name, condition_name

# Load class names
class_names = load_class_names(DATASET_DIR)


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
    prediction = model.predict(img_array, verbose=0)
    predicted_index = int(np.argmax(prediction, axis=1)[0])
    confidence = float(np.max(prediction)) * 100

    if predicted_index < len(class_names):
        predicted_class = class_names[predicted_index]
    else:
        predicted_class = "Unknown"

    plant_name, disease_name = format_prediction_label(predicted_class)

    st.subheader(" Prediction Result")
    st.write(f"**Plant:** {plant_name}")
    st.write(f"**Predicted Disease:** {disease_name}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Confidence-based decision
    if confidence < CONFIDENCE_THRESHOLD:
        st.warning(
            f" Low confidence prediction ({confidence:.2f}%). "
            "Please upload a clearer image or consult an expert."
        )
    else:
        if "healthy" in disease_name.lower():
            st.success(" The plant appears to be healthy")
        else:
            st.error(f"⚠️ Disease detected: {disease_name}")


# FOOTER

st.markdown("---")
st.caption(
    "Model: MobileNetV2 (Transfer Learning + Fine-Tuning) | "
    "Project: Plant Disease Detection Using Deep Learning"
)
