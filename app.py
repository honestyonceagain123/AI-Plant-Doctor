import streamlit as st

# ✅ Must be only once and before everything
st.set_page_config(page_title="🍃 AI Plant Doctor", layout="wide")

import torch
from torchvision import models, transforms
from PIL import Image
from deep_translator import GoogleTranslator
import os


# -----------------------------
# 🌱 LOAD MODEL (optional placeholder)
# -----------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 15)

    model_path = "plant_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        st.warning("⚠️ Model file not found — please add plant_model.pth to your folder.")
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# -----------------------------
# 🌿 TREATMENT INFO
# -----------------------------
treatment_info = {
    "Pepper__bell___Bacterial_spot": "Remove infected leaves immediately. Avoid overhead watering. Spray copper-based fungicide every 7–10 days. Rotate crops yearly.",
    "Pepper__bell___healthy": "Your pepper plant is healthy! Maintain good watering and monitor regularly.",
    "Potato___Early_blight": "Spray Mancozeb or Chlorothalonil fungicide. Remove infected leaves and ensure proper air circulation.",
    "Potato___healthy": "Your potato plant is healthy! Continue regular watering and sunlight.",
    "Potato___Late_blight": "Remove infected leaves immediately. Use Metalaxyl or Chlorothalonil fungicides. Avoid overhead irrigation.",
    "Tomato_Bacterial_spot": "Use copper-based bactericides. Remove infected leaves and ensure spacing between plants.",
    "Tomato_Early_blight": "Spray Mancozeb or Chlorothalonil. Prune affected leaves and mulch soil.",
    "Tomato_healthy": "Your tomato plant is healthy! Maintain consistent watering and sunlight.",
    "Tomato_Late_blight": "Remove infected plants. Use fungicides like Mancozeb. Avoid wetting leaves while watering.",
    "Tomato_Leaf_Mold": "Improve ventilation, reduce humidity. Apply sulfur-based fungicides.",
    "Tomato_Septoria_leaf_spot": "Remove infected debris. Spray copper fungicide. Rotate crops annually.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Spray neem oil. Increase humidity. Avoid nitrogen-rich fertilizer.",
    "Tomato__Target_Spot": "Remove affected leaves. Apply Mancozeb. Improve air circulation.",
    "Tomato__Tomato_mosaic_virus": "No cure. Remove infected plants. Disinfect tools. Use resistant varieties.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Control whiteflies. Remove infected plants. Use resistant seeds."
}

# -----------------------------
# 🌍 LANGUAGES
# -----------------------------
languages = [
    "English", "Hindi", "Marathi", "Gujarati", "Tamil", "Telugu",
    "Bengali", "Kannada", "Punjabi", "Urdu", "Odia", "Malayalam", "Nepali"
]

lang_codes = {
    "English": "en", "Hindi": "hi", "Marathi": "mr", "Gujarati": "gu",
    "Tamil": "ta", "Telugu": "te", "Bengali": "bn", "Kannada": "kn",
    "Punjabi": "pa", "Urdu": "ur", "Odia": "or", "Malayalam": "ml", "Nepali": "ne"
}

# -----------------------------
# 🎨 STYLING
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #e8f5e9;
    font-family: 'Poppins', sans-serif;
}
h1, h2, h3, h4 {
    font-weight: 800 !important;
}
h1 {
    text-align: center;
    color: #b9f6ca;
    text-shadow: 1px 1px 2px #1b5e20;
    font-size: 2.8rem;
}
.upload-box, .result-card {
    background-color: #1e1e1e;
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 0 15px rgba(0,255,100,0.15);
    margin-top: 20px;
}
.stButton>button {
    background: linear-gradient(90deg, #43a047, #66bb6a);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 1.5rem;
    font-size: 1rem;
    font-weight: 700;
}
.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #81c784, #2e7d32);
}
footer, #MainMenu, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 🏠 HEADER
# -----------------------------
col1, col2 = st.columns([7, 3])
with col2:
    selected_lang = st.selectbox("🌐 Choose Language", languages, index=0)

logo_path = "html/logo.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=100)

main_title = "🍃 AI Plant Doctor"
subtitle = "Upload a leaf image to detect plant disease and get treatment suggestions instantly."

target_lang = lang_codes.get(selected_lang, "en")
if selected_lang != "English":
    try:
        translator = GoogleTranslator(source="auto", target=target_lang)
        main_title = translator.translate(main_title)
        subtitle = translator.translate(subtitle)
    except:
        pass

st.markdown(f"<h1>{main_title}</h1>", unsafe_allow_html=True)
st.markdown(f"<h4 style='text-align:center;font-weight:600;'>{subtitle}</h4>", unsafe_allow_html=True)

# -----------------------------
# 📤 IMAGE UPLOAD + PREDICTION + TREATMENT
# -----------------------------
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
upload_label = "📸 Upload a leaf image"
if selected_lang != "English":
    try:
        upload_label = translator.translate(upload_label)
    except:
        pass

uploaded_file = st.file_uploader(upload_label, type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        # Load image safely
        with Image.open(uploaded_file) as img:
            image = img.convert("RGB")
        st.image(image, caption="🖼️ Uploaded Leaf", width=700)


        # Preprocess for model
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Prediction
        with st.spinner("🔍 Analyzing the image..."):
            outputs = model(img_tensor)
            _, preds = torch.max(outputs, 1)

        class_names = list(treatment_info.keys())
        predicted_class = class_names[preds.item()]
        formatted_class = predicted_class.replace("__", " ").replace("_", " ").title()
        treatment_text = treatment_info[predicted_class]

        disease_label = "Disease Detected"
        treatment_label = "Recommended Treatment"

        if selected_lang != "English":
            try:
                disease_label = translator.translate(disease_label)
                treatment_label = translator.translate(treatment_label)
                formatted_class = translator.translate(formatted_class)
                treatment_text = translator.translate(treatment_text)
            except:
                st.warning("⚠️ Translation temporarily unavailable. Showing English text.")

                      # Display result with larger, readable treatment
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        # 🌿 Disease Detected
        st.markdown(f"<h3 style='color:#1b5e20; font-weight:800;'>🌿 {disease_label}:</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='font-size:1.5rem; font-weight:700; color:#000000; background-color:#a5d6a7; padding:15px; border-radius:10px;'>"
            f"{formatted_class}</p>",
            unsafe_allow_html=True
        )

        # 💊 Recommended Treatment (convert to bullet points)
        bullet_points = ""
        for sentence in treatment_text.split("."):
            sentence = sentence.strip()
            if sentence:
                bullet_points += f"<li>{sentence}.</li>"

        st.markdown(f"<h3 style='margin-top:15px; color:#2e7d32; font-weight:800;'>💊 {treatment_label}:</h3>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <ul style='font-size:1.2rem; font-weight:600; line-height:1.8; color:#000000;
                       background-color:#c8e6c9; padding:15px; border-radius:10px;'>
                {bullet_points}
            </ul>
            """,
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)



    except Exception as e:
        st.error(f"⚠️ Failed to process the uploaded image. Details: {e}")

# -----------------------------
# 💬 CHATBOT INTEGRATION (Dialogflow)
# -----------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h3 style='color:#81c784; text-align:center;'>Chat with our AI Assistant</h3>", unsafe_allow_html=True)

chatbot_html = """
<script src="https://www.gstatic.com/dialogflow-console/fast/messenger/bootstrap.js?v=1"></script>
<df-messenger
  intent="WELCOME"
  chat-title="Education_App_Bot"
  agent-id="4aaa36e0-4df6-4743-aa90-651e33adc562"
  language-code="en"
></df-messenger>
"""
st.components.v1.html(chatbot_html, height=500, scrolling=True)




# -----------------------------
# 💚 FOOTER
# -----------------------------
footer = "Made with 💚 by Deven Aggarwal & Garvit Mina"
if selected_lang != "English":
    try:
        footer = translator.translate(footer)
    except:
        pass
st.markdown(f"<center><p style='color:#81c784;font-weight:700;'>{footer}</p></center>", unsafe_allow_html=True)
