import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load trained model (replace with your .h5 file path)
model = load_model('plant_disease_model.h5')

# Mapping class indices to labels
class_labels = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy'
    # Add all classes here based on your training dataset
}

st.title("🌱 Plant Disease Detection")
st.write("Upload a plant leaf image and get instant disease prediction.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Leaf Image', use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    disease = class_labels.get(class_idx, "Unknown")

    st.success(f"Predicted Disease: {disease}")
    st.write(f"Confidence: {prediction[0][class_idx]*100:.2f}%")
