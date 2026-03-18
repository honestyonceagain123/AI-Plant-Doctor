# app.py — AI Plant Doctor (Dark Theme, Center Big Login, Floating Username)
# Requirements: streamlit, torch, torchvision, gTTS, deep_translator, google-generativeai, requests, pandas, pillow, numpy, opencv-python-headless

import os
import sys

# 🔥 FIX: Use environment variables to force PyTorch
os.environ["DISABLE_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import re
import io
import time
import requests
import streamlit as st
from PIL import Image
import numpy as np
import cv2

import torch
import torch.nn as nn
from torchvision import models, transforms
from deep_translator import GoogleTranslator
import google.generativeai as genai  # Gemini API

from gtts import gTTS, lang as gtts_langs
import pandas as pd

import streamlit.components.v1 as components

# --------------------------
# Page config
# --------------------------
st.set_page_config(page_title="AI Plant Doctor", page_icon="🌱", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# KHATARNAK DARK THEME CSS 🔥
# ==========================================
st.markdown(
    """
    <style>
    /* Force Dark Theme Backgrounds */
    .stApp {
        background-color: #0f172a; /* Slate 900 */
        color: #f8fafc; /* Slate 50 */
    }
    
    [data-testid="stHeader"] {
        background-color: transparent;
    }

    [data-testid="stSidebar"] {
        background-color: #1e293b; /* Slate 800 */
        border-right: 1px solid #334155;
    }

    /* Primary Buttons (Green Gradient) */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.3);
        transition: all 0.2s;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.4);
        border: none;
    }

    /* Hide Streamlit Branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Login Box Container Styling */
    .login-container {
        background-color: #1e293b;
        padding: 40px;
        border-radius: 16px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        border: 1px solid #334155;
    }

    /* Floating Welcome Banner Animation (5 Sec) */
    @keyframes slideDownUp {
        0% { top: -100px; opacity: 0; transform: translateX(-50%) scale(0.9); }
        10% { top: 30px; opacity: 1; transform: translateX(-50%) scale(1); }
        85% { top: 30px; opacity: 1; transform: translateX(-50%) scale(1); }
        100% { top: -100px; opacity: 0; transform: translateX(-50%) scale(0.9); }
    }
    .welcome-float {
        position: fixed;
        top: -100px;
        left: 50%;
        transform: translateX(-50%);
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 16px 40px;
        border-radius: 50px;
        font-size: 20px;
        font-weight: bold;
        box-shadow: 0 15px 35px rgba(16,185,129,0.4);
        z-index: 9999999;
        animation: slideDownUp 5s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
        pointer-events: none;
        border: 2px solid rgba(255,255,255,0.2);
    }
    
    /* Text Inputs Dark */
    .stTextInput input {
        background-color: #334155 !important;
        color: #f8fafc !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
    }
    .stTextInput input:focus {
        border-color: #10b981 !important;
        box-shadow: 0 0 0 1px #10b981 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Session States
# --------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user_email" not in st.session_state:
    st.session_state["user_email"] = ""
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "just_logged_in" not in st.session_state:
    st.session_state["just_logged_in"] = False

# --------------------------
# Languages mapping
# --------------------------
LANGUAGES = {
    "English": "en", "Hindi (हिन्दी)": "hi", "Tamil (தமிழ்)": "ta", "Telugu (తెలుగు)": "te",
    "Bengali (বাংলা)": "bn", "Gujarati (ગુજરાતી)": "gu", "Punjabi (ਪੰਜਾਬੀ)": "pa",
    "Marathi (मराठी)": "mr", "Urdu (اردو)": "ur", "French (Français)": "fr",
    "German (Deutsch)": "de", "Spanish (Español)": "es", "Chinese (中文)": "zh-cn", "Japanese (日本語)": "ja"
}

# --------------------------
# Firebase Auth Functions (REST API)
# --------------------------
def firebase_login(email, password, api_key):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
    data = {"email": email, "password": password, "returnSecureToken": True}
    r = requests.post(url, json=data)
    return r.json()

def firebase_signup(email, password, api_key):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={api_key}"
    data = {"email": email, "password": password, "returnSecureToken": True}
    r = requests.post(url, json=data)
    return r.json()

# --------------------------
# Model loaders (cached)
# --------------------------
@st.cache_resource
def load_plant_detector():
    model = models.resnet18(weights=None) 
    model.fc = nn.Linear(model.fc.in_features, 2)
    try:
        checkpoint = torch.load("plant_detector_final.pth", map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(DEVICE)
        model.eval()
        return model
    except FileNotFoundError:
        return None

@st.cache_resource
def load_classifier(weight_path="best_plant_model.pth"):
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 240)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Classifier weight missing: {weight_path}")
    state = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

@st.cache_data
def get_class_names(train_dir="dataset/train"):
    if not os.path.isdir(train_dir):
        return []
    return sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

# --------------------------
# Image Validation Helpers
# --------------------------
def is_clear_leaf_image(pil_img):
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 80: return False
    return True

def has_enough_green(image: Image.Image):
    img = np.array(image)
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    green_pixels = (g > r + 15) & (g > b + 15)
    green_ratio = np.sum(green_pixels) / green_pixels.size
    return green_ratio > 0.05

def is_plant_image(image, model):
    if model is None: return True, 100.0
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_t = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(img_t)
        probs = torch.softmax(logits, dim=1)[0]
    plant_prob = probs[1].item()
    entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
    if plant_prob < 0.95 or entropy > 0.25 or not has_enough_green(image):
        return False, plant_prob * 100
    return True, plant_prob * 100

# --------------------------
# Helpers & TTS cleaning
# --------------------------
def clean_text_for_tts(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`.+?`", " ", text)
    text = re.sub(r"^[\-\*\+\>\#]+\s*", " ", text, flags=re.M)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"(\/{2,}|\\{2,}|\*{1,}|\_{1,}|\={1,}|\|{1,}|\[{1,}|\]{1,})", " ", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 0 and text[-1] not in ".!?":
        text = text + "."
    return text

def generate_tts_bytes(text: str, lang_code: str = "en") -> bytes:
    cleaned = clean_text_for_tts(text)
    try: supported = gtts_langs.tts_langs()
    except Exception: supported = {"en":"English"}
    
    if lang_code not in supported:
        if lang_code.startswith("zh"): lang_code = "zh-cn" if "zh-cn" in supported else "zh"
        elif lang_code == "pa" and "pa" not in supported: lang_code = "hi" if "hi" in supported else "en"
        else: lang_code = "en"
        
    audio_buf = io.BytesIO()
    try:
        tts = gTTS(text=cleaned, lang=lang_code)
        tts.write_to_fp(audio_buf)
        audio_buf.seek(0)
        return audio_buf.read()
    except Exception:
        try:
            tts = gTTS(text=cleaned, lang="en")
            audio_buf = io.BytesIO()
            tts.write_to_fp(audio_buf)
            audio_buf.seek(0)
            return audio_buf.read()
        except Exception: return b""

# --------------------------
# Gemini AI treatment
# --------------------------
def generate_treatment_with_ai(disease_name: str) -> str:
    if "GEMINI_API_KEY" not in st.secrets:
        return "⚠️ GEMINI_API_KEY missing from Streamlit secrets."
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""You are an expert agronomist AI. A farmer's leaf is diagnosed as '{disease_name}'.
        Provide: 1. A short cause summary. 2. Clear actionable treatment steps. 3. Preventive tips. Keep it simple."""
        return model.generate_content(prompt).text.strip()
    except Exception as e:
        return f"⚠️ AI generation failed: {e}"

# --------------------------
# OpenWeather free helpers
# --------------------------
def geocode_city_to_coords(city: str, api_key: str):
    try:
        r = requests.get("http://api.openweathermap.org/geo/1.0/direct", params={"q": f"{city},IN", "limit": 1, "appid": api_key}, timeout=12)
        data = r.json()
        if not data: return None
        return float(data[0]["lat"]), float(data[0]["lon"]), data[0].get("name", city)
    except Exception: return None

def get_current_weather(city: str, api_key: str):
    try:
        return requests.get("https://api.openweathermap.org/data/2.5/weather", params={"q": f"{city},IN", "units": "metric", "appid": api_key}, timeout=10).json()
    except Exception: return None

def get_forecast_3h(city: str, api_key: str):
    try:
        return requests.get("https://api.openweathermap.org/data/2.5/forecast", params={"q": f"{city},IN", "units": "metric", "appid": api_key}, timeout=10).json()
    except Exception: return None

def get_todays_forecast_from_3h(forecast_json):
    if not forecast_json or "list" not in forecast_json: return []
    today = time.strftime("%Y-%m-%d", time.localtime())
    items = []
    for item in forecast_json["list"]:
        dt_txt = item.get("dt_txt", "")
        if dt_txt.startswith(today):
            items.append({"time": dt_txt, "temp": item.get("main", {}).get("temp"), "feels_like": item.get("main", {}).get("feels_like"), "humidity": item.get("main", {}).get("humidity"), "desc": item.get("weather", [{}])[0].get("description", "")})
    return items

def assess_weather_risk_with_ai(daily_forecasts: list, location_name: str = "", gemini_key: str = None):
    summary_lines = [f"{d.get('time')}: {d.get('desc')}, temp {d.get('temp')}°C, hum {d.get('humidity')}%" for d in daily_forecasts]
    summary_text = "\n".join(summary_lines[:12])
    if gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            prompt = f"Expert agriculture short risk assessment for {location_name} based on: {summary_text}. Give Risk level (Low/Mod/High) for fungal/pest and 3 quick tips."
            return model.generate_content(prompt).text.strip()
        except Exception: pass
    return "Risk Summary: Moderate.\nRecommendations: Ensure drainage, monitor crops."

# --------------------------
# Prediction Helper
# --------------------------
def predict(image: Image.Image, model, class_names):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, 1)
    idx = preds.item()
    name = class_names[idx] if class_names and idx < len(class_names) else f"class_{idx}"
    return name, float(conf.item() * 100.0)


# ==========================
# CENTERED BIG LOGIN UI
# ==========================
def render_auth_page():
    # Hide sidebar when logged out
    st.markdown("""<style>[data-testid="collapsedControl"] {display: none;}</style>""", unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #10b981; font-size: 4em; font-weight: 900;'>AI Plant Doctor</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #94a3b8; margin-bottom: 40px;'>Your Smart Agricultural Assistant</h4>", unsafe_allow_html=True)

    if "FIREBASE_API_KEY" not in st.secrets:
        st.error("⚠️ FIREBASE_API_KEY missing in secrets.toml")
        return
    api_key = st.secrets["FIREBASE_API_KEY"]

    # Use columns to center the login box and make it comfortably wide
    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🔑 Secure Login", "📝 Create Account"])
        
        with tab1:
            st.markdown("### Welcome Back")
            login_email = st.text_input("Email", key="login_email", placeholder="farmer@example.com")
            login_password = st.text_input("Password", type="password", key="login_password", placeholder="••••••••")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Login", use_container_width=True):
                if login_email and login_password:
                    with st.spinner("Authenticating..."):
                        res = firebase_login(login_email, login_password, api_key)
                        if "error" in res:
                            st.error(f"❌ {res['error'].get('message', 'Login failed')}")
                        else:
                            st.session_state["logged_in"] = True
                            st.session_state["user_email"] = res["email"]
                            st.session_state["username"] = res["email"].split("@")[0].capitalize()
                            st.session_state["just_logged_in"] = True
                            st.rerun()
                else:
                    st.warning("Enter email and password.")
                    
        with tab2:
            st.markdown("### Join AI Plant Doctor")
            signup_email = st.text_input("Email", key="signup_email", placeholder="farmer@example.com")
            signup_password = st.text_input("Password", type="password", key="signup_password", placeholder="Min 6 characters")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Sign Up", use_container_width=True):
                if signup_email and len(signup_password) >= 6:
                    with st.spinner("Creating profile..."):
                        res = firebase_signup(signup_email, signup_password, api_key)
                        if "error" in res:
                            st.error(f"❌ {res['error'].get('message', 'Signup failed')}")
                        else:
                            st.success("✅ Account created successfully! Please switch to the Login tab.")
                else:
                    st.warning("Valid email and 6+ char password required.")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# UNLOCKED MAIN APP
# ==========================
def main_app():
    # Show sidebar control when logged in
    st.markdown("""<style>[data-testid="collapsedControl"] {display: block;}</style>""", unsafe_allow_html=True)

    # 5-SECOND FLOATING USERNAME POPUP
    if st.session_state.get("just_logged_in"):
        st.session_state["just_logged_in"] = False
        st.markdown(
            f"""
            <div class="welcome-float">
                👋 Welcome, <span style="color:#a7f3d0;">{st.session_state['username']}</span>! Let's heal some plants.
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------
    # Logged-In Sidebar
    # --------------------------
    st.sidebar.markdown(f"### 👤 Profile")
    st.sidebar.info(f"**{st.session_state['username']}**\n\n{st.session_state['user_email']}")
    
    if st.sidebar.button("🚪 Logout"):
        st.session_state["logged_in"] = False
        st.session_state["user_email"] = ""
        st.session_state["username"] = ""
        st.rerun()
        
    st.sidebar.markdown("---")
    if DEVICE == "cuda":
        try: name = torch.cuda.get_device_name(0)
        except: name = "CUDA GPU"
        st.sidebar.success(f"🚀 Powered by: {name}")
    else:
        st.sidebar.warning("⚡ Running on CPU.")
    st.sidebar.markdown("---")

    # --------------------------
    # Weather Module in Sidebar
    # --------------------------
    if "weather_info" not in st.session_state: st.session_state.weather_info = None
    if "manual_city" not in st.session_state: st.session_state.manual_city = ""

    st.sidebar.header("🌦️ Field Weather")
    st.session_state.manual_city = st.sidebar.text_input("Enter City (e.g., Mumbai)", value=st.session_state.manual_city)

    if st.sidebar.button("Get Weather"):
        city = st.session_state.manual_city.strip()
        if city and "OPENWEATHER_KEY" in st.secrets:
            coords = geocode_city_to_coords(city, st.secrets["OPENWEATHER_KEY"])
            city_name = coords[2] if coords else city
            cur = get_current_weather(city_name, st.secrets["OPENWEATHER_KEY"])
            forecast = get_forecast_3h(city_name, st.secrets["OPENWEATHER_KEY"])
            if cur and cur.get("cod") != 401:
                st.session_state.weather_info = {"city": city_name, "current": cur, "forecast3h": forecast}
                st.sidebar.success(f"Weather updated: {city_name}")
            else: st.sidebar.error("Error fetching weather.")

    if st.session_state.get("weather_info"):
        info = st.session_state["weather_info"]
        if info.get("current"):
            cur = info["current"]
            st.sidebar.markdown(f"**{info['city']}** | 🌡️ {cur['main']['temp']}°C | 💧 {cur['main']['humidity']}%")
            st.sidebar.caption(cur["weather"][0]["description"].title())

    # --------------------------
    # Main App UI
    # --------------------------
    col_main, col_lang = st.columns([3, 1])
    with col_lang:
        selected_label = st.selectbox("🌐 Translation & Voice", list(LANGUAGES.keys()), index=list(LANGUAGES.keys()).index("English"))
        target_lang_code = LANGUAGES.get(selected_label, "en")
        translator_ui = GoogleTranslator(source="auto", target=target_lang_code)

    def t(text: str) -> str:
        try: return text if selected_label == "English" else translator_ui.translate(text)
        except Exception: return text

    with col_main:
        st.title(t("🩺 AI Plant Doctor Core"))
        st.markdown(t("Upload a close-up photo of the affected leaf for instant diagnosis and AI-generated treatment."))

    st.markdown("---")
    uploaded_file = st.file_uploader(t("📤 Upload a leaf image"), type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        # Ensure image takes up reasonable space without being huge
        col_img, _ = st.columns([1, 1])
        with col_img:
            st.image(image, caption=t("Uploaded Image"), use_column_width=True)

        if st.button(t("🔍 Analyze Plant Health"), use_container_width=True):
            if not is_clear_leaf_image(image):
                st.error(t("❌ Image is blurry or not a leaf"))
            else:
                detector = load_plant_detector()
                with st.spinner(t("Scanning leaf biology...")):
                    plant_ok, plant_conf = is_plant_image(image, detector)

                if not plant_ok or plant_conf < 90:
                    st.error(t("❌ Uploaded image is NOT a clear plant"))
                    st.warning(t("Please upload a close-up leaf with plain background"))
                else:
                    st.success(t(f"✅ Plant verified ({plant_conf:.2f}%)"))

                    try:
                        model = load_classifier()
                        class_names = get_class_names()
                        with st.spinner(t("Running Neural Network...")):
                            prediction, confidence = predict(image, model, class_names)
                            
                            st.success(f"🌿 {t('Diagnosis')}: {t(prediction)} (Conf: {confidence:.2f}%)")

                            if str(prediction).lower() == "healthy":
                                st.balloons()
                                st.success(t("Great news! The plant looks healthy. Keep up the good work!"))
                            else:
                                with st.spinner(t("Generating AI Treatment Protocol...")):
                                    treatment_en = generate_treatment_with_ai(prediction)
                                    translated_treatment = treatment_en if selected_label == "English" else GoogleTranslator(source="auto", target=target_lang_code).translate(treatment_en)

                                st.subheader(t("AI-Generated Treatment Plan"))
                                st.info(translated_treatment)

                                with st.spinner(t("Generating Voice Assistant...")):
                                    treatment_audio = generate_tts_bytes(translated_treatment, lang_code=target_lang_code)
                                if treatment_audio and len(treatment_audio) > 10:
                                    st.audio(treatment_audio, format="audio/mp3")
                                    st.download_button(label=t("⬇️ Download Audio"), data=treatment_audio, file_name="treatment.mp3", mime="audio/mpeg")
                    except Exception as e:
                        st.error(f"Engine Error: {e}")

    # Weather Risk Section
    st.markdown("---")
    st.header(t("🌦️ AI Weather Risk Analysis"))
    if not st.session_state.get("weather_info"):
        st.info(t("Set your City in the sidebar to get weather-based disease risk predictions."))
    else:
        info = st.session_state["weather_info"]
        todays_items = get_todays_forecast_from_3h(info.get("forecast3h"))
        if todays_items:
            with st.spinner(t("Analyzing weather conditions...")):
                raw_risk = assess_weather_risk_with_ai(todays_items, info.get("city",""), st.secrets.get("GEMINI_API_KEY"))
                translated_risk = raw_risk if selected_label == "English" else GoogleTranslator(source="auto", target=target_lang_code).translate(raw_risk)
            st.warning(translated_risk)
        else:
            st.info(t("Detailed forecast unavailable currently."))

    st.markdown("---")
    st.caption("© 2026 AI Plant Doctor — Empowering Farmers with AI 🌾")

    # --------------------------
    # Floating Chat Widget (Protected)
    # --------------------------
    chatbot_url = "https://light-yagami980.diaflow.app/public-chat/RGMNeOWpcT"
    floating_widget = f"""
    <style>
    #floatingChatContainer {{ position: fixed; bottom: 0; right: 0; z-index: 999999999; pointer-events: none; }}
    #chatButton {{ width: 65px; height: 65px; background: linear-gradient(135deg, #10b981, #059669); color: white; border-radius: 50%; border: none; font-size: 30px; cursor: pointer; margin: 20px; box-shadow: 0px 4px 15px rgba(16,185,129,0.5); pointer-events: auto; transition: transform 0.3s; }}
    #chatButton:hover {{ transform: scale(1.1); }}
    #chatFrame {{ width: 360px; height: 480px; border-radius: 14px; border: none; display: none; margin-right: 20px; margin-bottom: 95px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); pointer-events: auto; }}
    </style>
    <div id="floatingChatContainer">
        <iframe id="chatFrame" src="{chatbot_url}"></iframe>
        <button id="chatButton">💬</button>
    </div>
    <script>
    const btn = document.getElementById("chatButton");
    const frame = document.getElementById("chatFrame");
    btn.onclick = function() {{ frame.style.display = frame.style.display === "none" ? "block" : "none"; }};
    </script>
    """
    components.html(floating_widget, height=0, width=0)

# ==========================
# APP ROUTER
# ==========================
if st.session_state["logged_in"]:
    main_app()
else:
    render_auth_page()
