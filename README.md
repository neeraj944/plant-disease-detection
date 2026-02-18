#  Plant Disease Detection Using Deep Learning

This project implements a **Plant Leaf Disease Detection System** using **Deep Learning (CNN)** with **Transfer Learning and Fine-Tuning**.  
The system allows users to upload a leaf image and automatically predicts whether the plant is **healthy or affected by a disease**, along with a confidence score.

The trained deep learning model is deployed using **Streamlit**, making the project interactive and user-friendly.

---

##  Project Overview

Plant diseases significantly affect agricultural productivity. Early and accurate detection of plant diseases can help farmers take preventive actions and reduce crop loss.

This project uses a **Convolutional Neural Network (CNN)** based on **MobileNetV2**, pretrained on ImageNet, and fine-tuned on plant leaf images from the **PlantVillage dataset**.

---

## Objectives

- To detect plant diseases from leaf images using deep learning  
- To use **transfer learning** for better accuracy and faster training  
- To build a simple **web application** for real-time disease prediction  
- To provide confidence-based prediction results  

---

##  Model & Techniques Used

- Model: MobileNetV2  
- Approach: 
  - Transfer Learning  
  - Fine-Tuning  
- Loss Function:** Categorical Crossentropy  
- Optimizer:** Adam  
- Framework:** TensorFlow / Keras  



##  Dataset

- Dataset: PlantVillage (subset)  
- Type: Image dataset  
- Number of Classes: 16  
- Plants Covered:
  - Tomato
  - Potato
  - Bell Pepper
  - Healthy leaves included

Each class is stored in a separate directory for supervised learning.

---

##  System Workflow

1. User uploads a leaf image through the web interface  
2. Image is resized and normalized  
3. Image is passed to the trained CNN model  
4. Model predicts the disease class  
5. Confidence score is calculated  
6. Result is displayed to the user  

---

## Web Application (Streamlit)

The project includes a **Streamlit-based web application** with the following features:

- Upload leaf images (JPG / PNG)
- Display uploaded image
- Predict disease name
- Show confidence percentage
- Indicate whether the plant is healthy or diseased
- Confidence threshold to avoid unreliable predictions

---


