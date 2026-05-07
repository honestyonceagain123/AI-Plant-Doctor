# app.py — AI Plant Doctor (Enterprise True Dark Theme, Plantix-style Image Overlay, Gen-Z Friendly Auth)
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
# ENTERPRISE TRUE DARK THEME CSS 🔥
# ==========================================
st.markdown(
    """
    <style>
    /* Force True Dark Backgrounds (Pitch Black & Dark Gray) */
    .stApp {
        background-color: #050505;
        background-image: radial-gradient(circle at 50% -20%, #151515 0%, #050505 100%);
        color: #f8fafc;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    [data-testid="stHeader"] { background-color: transparent; }
    [data-testid="stSidebar"] {
        background-color: #0d0d0d !important;
        border-right: 1px solid #1a1a1a;
    }
    
    /* Enterprise Primary Buttons (Sleek Green) */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff;
        border-radius: 12px;
        border: 1px solid #10b981;
        padding: 0.7rem 1.5rem;
        font-size: 16px;
        font-weight: 700;
        letter-spacing: 1px;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        text-transform: uppercase;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-4px) scale(1.01);
        box-shadow: 0 12px 30px rgba(16, 185, 129, 0.4);
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
        border-color: #34d399;
    }
    div.stButton > button:first-child:active {
        transform: translateY(0px);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    
    /* Interactive Login Box (Glassmorphism + Neon Edge) */
    .login-container {
        background: #111111;
        padding: 50px;
        border-radius: 20px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.8), inset 0 0 0 1px #222;
        border: 1px solid rgba(16, 185, 129, 0.15);
        transition: transform 0.4s ease, box-shadow 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    .login-container:hover {
        box-shadow: 0 30px 60px rgba(0, 0, 0, 0.9), 0 0 40px rgba(16, 185, 129, 0.1), inset 0 0 0 1px #333;
        transform: translateY(-5px);
        border-color: rgba(16, 185, 129, 0.4);
    }
    
    /* Segmented Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0a0a0a;
        border-radius: 12px;
        padding: 6px;
        gap: 8px;
        border-bottom: none !important;
        border: 1px solid #1f1f1f;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #666;
        border: none !important;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #10b981 !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    /* Custom File Uploader Zone */
    [data-testid="stFileUploadDropzone"] {
        background-color: #0f0f0f !important;
        border: 2px dashed #222 !important;
        border-radius: 16px !important;
        padding: 40px !important;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        background-color: #111814 !important;
        border-color: #10b981 !important;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.15);
    }
    
    /* Text Inputs & Areas (Ultra Smooth) */
    .stTextInput input, .stTextArea textarea {
        background-color: #161616 !important;
        color: #ffffff !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        font-size: 15px !important;
        transition: all 0.3s ease;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #10b981 !important;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2) !important;
        background-color: #1a1a1a !important;
    }
    .stTextInput input::placeholder, .stTextArea textarea::placeholder {
        color: #555 !important;
    }
    
    /* Floating Welcome Banner */
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
        font-size: 18px;
        font-weight: 700;
        box-shadow: 0 15px 35px rgba(16,185,129,0.5);
        z-index: 9999999;
        animation: slideDownUp 5s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
        pointer-events: none;
        border: 1px solid rgba(255,255,255,0.2);
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    /* Expander / Info Boxes */
    [data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid #10b981;
        background-color: rgba(16, 185, 129, 0.05);
    }
    .stExpander {
        background-color: #111;
        border-radius: 12px;
        border: 1px solid #222;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Session States & Auto-Login
# --------------------------
if "logged_in" not in st.session_state:
    # URL check karega ki user refresh karne se pehle logged in tha ya nahi
    if "auth_user" in st.query_params:
        st.session_state["logged_in"] = True
        st.session_state["user_email"] = st.query_params["auth_user"]
        st.session_state["username"] = st.query_params["auth_user"].split("@")[0].capitalize()
        st.session_state["just_logged_in"] = False
    else:
        st.session_state["logged_in"] = False
        st.session_state["user_email"] = ""
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
    if os.path.exists("classes.txt"):
        with open("classes.txt", "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    if not os.path.isdir(train_dir):
        return []
    return sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

# --------------------------
# Image Validation & Overlay Helpers (Plantix Style)
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

def draw_plantix_overlay(image: Image.Image, prediction: str, confidence: float):
    """
    Draws a professional, Plantix-style colored overlay on the uploaded image.
    """
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    h, w, _ = img_cv.shape
    
    # Determine Banner Color (Green for Healthy, Red for Disease)
    is_healthy = "healthy" in prediction.lower()
    banner_color = (30, 180, 50) if is_healthy else (30, 40, 220)  # BGR format
    
    # Calculate banner dimensions
    banner_h = max(int(h * 0.15), 50)
    
    # Create semi-transparent overlay
    overlay = img_cv.copy()
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), banner_color, -1)
    cv2.addWeighted(overlay, 0.85, img_cv, 0.15, 0, img_cv)
    
    # Setup Text
    font = cv2.FONT_HERSHEY_DUPLEX
    display_text = f"{prediction.replace('_', ' ').upper()} ({confidence:.1f}%)"
    
    # Dynamic Scaling to fit width
    font_scale = max(0.5, min(w, h) / 700.0)
    thickness = max(1, int(font_scale * 2.5))
    
    text_size = cv2.getTextSize(display_text, font, font_scale, thickness)[0]
    
    # Scale down if text is too wide for the image
    if text_size[0] > w * 0.95:
        font_scale = font_scale * ((w * 0.95) / text_size[0])
        thickness = max(1, int(font_scale * 2.5))
        text_size = cv2.getTextSize(display_text, font, font_scale, thickness)[0]
        
    # Center text coordinates
    text_x = (w - text_size[0]) // 2
    text_y = h - int(banner_h / 2) + (text_size[1] // 2)
    
    # Draw Shadow/Outline for better readability
    cv2.putText(img_cv, display_text, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    # Draw actual text (White)
    cv2.putText(img_cv, display_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    # Convert back to PIL Image
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

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
def generate_treatment_with_ai(disease_name: str, username: str, extra_notes: str = "") -> str:
    if "GEMINI_API_KEY" not in st.secrets:
        return "⚠️ GEMINI_API_KEY missing from Streamlit secrets."
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"You are an expert plant doctor talking directly to {username}. Start your response with 'Hello {username},'. A farmer's leaf is diagnosed with '{disease_name}'."
        if extra_notes:
            prompt += f" {username} also provided these extra details: '{extra_notes}'."
        prompt += "\nProvide: 1. A short cause summary. 2. Clear, easy treatment steps. 3. Quick preventive tips. Keep it friendly and simple."
        return model.generate_content(prompt).text.strip()
    except Exception as e:
        return f"⚠️ AI generation failed: {e}"

def generate_treatment_from_text(symptoms: str, username: str) -> str:
    if "GEMINI_API_KEY" not in st.secrets:
        return "⚠️ GEMINI_API_KEY missing from Streamlit secrets."
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""You are an expert plant doctor talking directly to {username}. Start your response with 'Hello {username},'. {username} does not have a photo but describes these symptoms: '{symptoms}'.
        Provide: 1. Possible causes/diseases based on the symptoms. 2. Clear, easy treatment steps. 3. Quick preventive tips. Keep it friendly and simple."""
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
            prompt = f"Act as a friendly farming assistant. For {location_name} with weather: {summary_text}. Give a very quick, simple risk level (Low/Mod/High) for crop diseases and 2 easy tips."
            return model.generate_content(prompt).text.strip()
        except Exception: pass
    return "Risk Level: Moderate. Ensure good drainage and keep an eye on your crops!"

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
# COOL GEN-Z LOGIN UI
# ==========================
def render_auth_page():
    st.markdown("""<style>[data-testid="collapsedControl"] {display: none;}</style>""", unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # 🔥 FIX: Solid Neon Green Color with Glowing shadow (Guaranteed to show in Streamlit)
    st.markdown("<h1 style='color: #10b981; text-shadow: 0 0 25px rgba(16, 185, 129, 0.4); text-align: center; font-size: 5.5rem; font-weight: 900; margin-bottom: 0px;'>🩺 AI Plant Doctor</h1>", unsafe_allow_html=True)
    
    st.markdown("<h4 style='text-align: center; color: #888; margin-bottom: 50px; font-weight: 500;'>Your Smart Farming Buddy 🌿</h4>", unsafe_allow_html=True)
    
    if "FIREBASE_API_KEY" not in st.secrets:
        st.error("⚠️ FIREBASE_API_KEY missing in secrets.toml")
        return
        
    api_key = st.secrets["FIREBASE_API_KEY"]
    col1, col2, col3 = st.columns([1, 1.3, 1])
    
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🔓 Login", "✨ Sign Up"])
        
        with tab1:
            st.markdown("<h3 style='margin-bottom: 25px; color:#fff;'>Welcome Back!</h3>", unsafe_allow_html=True)
            login_email = st.text_input("Email", key="login_email", placeholder="farmer@example.com")
            login_password = st.text_input("Password", type="password", key="login_password", placeholder="••••••••")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Let's Go! 🚀", use_container_width=True):
                if login_email and login_password:
                    with st.spinner("Logging you in..."):
                        res = firebase_login(login_email, login_password, api_key)
                        if "error" in res:
                            st.error(f"❌ Oops: {res['error'].get('message', 'Login failed')}")
                        else:
                            st.session_state["logged_in"] = True
                            st.session_state["user_email"] = res["email"]
                            st.session_state["username"] = res["email"].split("@")[0].capitalize()
                            st.session_state["just_logged_in"] = True
                            
                            # Refresh se bachne ke liye URL mein user info save karo
                            st.query_params["auth_user"] = res["email"]
                            
                            st.rerun()
                else:
                    st.warning("Please enter your email and password.")
                
                st.markdown("<div style='text-align: center; margin: 15px 0; color: #666;'>OR</div>", unsafe_allow_html=True)
                if st.button("🌐 Sign in with Google", key="google_login", use_container_width=True):
                    st.info("ℹ️ Google OAuth setup required in Firebase. For this demo, please use Email/Password.")
                    
        with tab2:
            st.markdown("<h3 style='margin-bottom: 25px; color:#fff;'>Join the Community</h3>", unsafe_allow_html=True)
            signup_email = st.text_input("Enter Email", key="signup_email", placeholder="farmer@example.com")
            signup_password = st.text_input("Create Password", type="password", key="signup_password", placeholder="At least 6 characters")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Account 🎉", use_container_width=True):
                if signup_email and len(signup_password) >= 6:
                    with st.spinner("Setting things up..."):
                        res = firebase_signup(signup_email, signup_password, api_key)
                        if "error" in res:
                            st.error(f"❌ Oops: {res['error'].get('message', 'Registration failed')}")
                        else:
                            st.success("✅ Awesome! Account created. Now just log in.")
                else:
                    st.warning("Needs a valid email and a 6+ char password.")
            
            st.markdown("<div style='text-align: center; margin: 15px 0; color: #666;'>OR</div>", unsafe_allow_html=True)
            if st.button("🌐 Sign up with Google", key="google_signup", use_container_width=True):
                st.info("ℹ️ Google OAuth setup required in Firebase. For this demo, please use Email/Password.")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# UNLOCKED MAIN APP
# ==========================
def main_app():
    st.markdown("""<style>[data-testid="collapsedControl"] {display: block;}</style>""", unsafe_allow_html=True)
    
    if st.session_state.get("just_logged_in"):
        st.session_state["just_logged_in"] = False
        st.markdown(
            f"""
            <div class="welcome-float">
                <span>🌿</span> Welcome back, <span style="color:#a7f3d0;">{st.session_state['username']}</span>!
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------
    # Logged-In Sidebar (Removed Profile & Logout)
    # --------------------------
    if DEVICE == "cuda":
        try: name = torch.cuda.get_device_name(0)
        except: name = "CUDA GPU"
        st.sidebar.success(f"🚀 Running fast on: {name}")
    else:
        st.sidebar.warning("⚡ Running on CPU")
    st.sidebar.markdown("---")

    # --------------------------
    # Weather Module in Sidebar
    # --------------------------
    if "weather_info" not in st.session_state: st.session_state.weather_info = None
    if "manual_city" not in st.session_state: st.session_state.manual_city = ""
    st.sidebar.header("🌦️ Weather Check")
    st.session_state.manual_city = st.sidebar.text_input("Enter City (India)", value=st.session_state.manual_city, placeholder="e.g. Pune")
    if st.sidebar.button("Get Weather"):
        city = st.session_state.manual_city.strip()
        if city and "OPENWEATHER_KEY" in st.secrets:
            coords = geocode_city_to_coords(city, st.secrets["OPENWEATHER_KEY"])
            city_name = coords[2] if coords else city
            cur = get_current_weather(city_name, st.secrets["OPENWEATHER_KEY"])
            forecast = get_forecast_3h(city_name, st.secrets["OPENWEATHER_KEY"])
            if cur and cur.get("cod") != 401:
                st.session_state.weather_info = {"city": city_name, "current": cur, "forecast3h": forecast}
                st.sidebar.success(f"Weather fetched for {city_name}!")
            else: st.sidebar.error("Couldn't fetch weather right now.")
            
    if st.session_state.get("weather_info"):
        info = st.session_state["weather_info"]
        if info.get("current"):
            cur = info["current"]
            st.sidebar.markdown(f"**{info['city']}** | 🌡️ {cur['main']['temp']}°C | 💧 {cur['main']['humidity']}%")
            st.sidebar.caption(cur["weather"][0]["description"].title())

    # --------------------------
    # Main App UI Header & Settings Bar
    # --------------------------
    col_main, col_settings = st.columns([3, 1])
    
    with col_settings:
        with st.popover("⚙️ Settings", use_container_width=True):
            st.markdown(f"### 👤 {st.session_state['username']}")
            st.caption(st.session_state['user_email'])
            st.markdown("---")
            
            # Moved Language Selector inside Settings
            selected_label = st.selectbox("🌐 Choose Language", list(LANGUAGES.keys()), index=list(LANGUAGES.keys()).index("English"))
            target_lang_code = LANGUAGES.get(selected_label, "en")
            
            st.markdown("---")
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state["logged_in"] = False
                st.session_state["user_email"] = ""
                st.session_state["username"] = ""
                if "auth_user" in st.query_params:
                    del st.query_params["auth_user"]
                st.rerun()

    translator_ui = GoogleTranslator(source="auto", target=target_lang_code)
        
    def t(text: str) -> str:
        try: return text if selected_label == "English" else translator_ui.translate(text)
        except Exception: return text
        
    with col_main:
        title_text = t('AI Plant Doctor')
        # 🔥 FIX: Solid Neon Green Color with Glowing shadow
        st.markdown(
            f"<h1 style='color: #10b981; text-shadow: 0 0 20px rgba(16, 185, 129, 0.4); font-size: 4rem; font-weight: 900; margin-bottom: 0px; padding-bottom: 10px;'>🩺 {title_text}</h1>", 
            unsafe_allow_html=True
        )
        st.markdown(f"<p style='color: #94a3b8; font-size: 1.2rem; margin-top: -15px;'>{t('Hello')} <span style='color: #10b981; font-weight: bold;'>{st.session_state['username']}</span>, {t('diagnose plant diseases instantly using AI.')}</p>", unsafe_allow_html=True)
    st.markdown("---")

    # ==========================
    # INPUT TABS (With Photo, Without Photo & Type Box)
    # ==========================
    tab_image, tab_text = st.tabs([t("📷 Diagnose by Photo"), t("✍️ Diagnose by Symptoms (No Photo)")])

    # --------------------------
    # TAB 1: Image Diagnosis (Upload or Camera + Text Box)
    # --------------------------
    with tab_image:
        input_col, opt_col = st.columns([2, 1])
        
        with input_col:
            img_source = st.radio(t("Select Image Source:"), [t("Upload File"), t("Use Camera")], horizontal=True)
            img_data = None
            
            if img_source == t("Upload File"):
                img_data = st.file_uploader(t("📤 Upload Leaf Image"), type=["jpg", "jpeg", "png"])
            else:
                img_data = st.camera_input(t("📸 Take Photo of Leaf"))
                
        with opt_col:
            extra_symptoms = st.text_area(t("Optional: Describe any other details (e.g. yellow spots, bugs seen)"), height=150)
            
        if img_data:
            image = Image.open(img_data).convert("RGB")
            col_img, _ = st.columns([1, 1])
            img_display = col_img.empty()
            img_display.image(image, caption=t("Your Photo"), use_column_width=True)
            
            if st.button(t("🔍 Analyze Plant"), use_container_width=True):
                if not is_clear_leaf_image(image):
                    st.error(t("❌ Image is blurry. Please upload a clear photo."))
                else:
                    detector = load_plant_detector()
                    with st.spinner(t("Checking image...")):
                        plant_ok, plant_conf = is_plant_image(image, detector)
                        
                    if not plant_ok or plant_conf < 90:
                        st.error(t("❌ This doesn't look like a clear plant leaf."))
                        st.warning(t("Try taking a closer photo of just the leaf."))
                    else:
                        try:
                            model = load_classifier()
                            class_names = get_class_names()
                            with st.spinner(t("Analyzing leaf...")):
                                prediction, confidence = predict(image, model, class_names)
                                
                                # PLANTIX-STYLE DYNAMIC IMAGE OVERLAY
                                annotated_img = draw_plantix_overlay(image, prediction, confidence)
                                img_display.image(annotated_img, caption=t("AI Result Overlay"), use_column_width=True)
                                
                                if str(prediction).lower() == "healthy":
                                    st.balloons()
                                    st.success(t("Yay! Your plant looks perfectly healthy. 🎉"))
                                else:
                                    with st.spinner(t("Finding the best treatment...")):
                                        # Pass the "type box" info and username to the AI too
                                        treatment_en = generate_treatment_with_ai(prediction, st.session_state['username'], extra_notes=extra_symptoms)
                                        translated_treatment = treatment_en if selected_label == "English" else GoogleTranslator(source="auto", target=target_lang_code).translate(treatment_en)
                                        
                                    st.subheader(t("AI Treatment Plan 💡"))
                                    st.info(translated_treatment)
                                    
                                    with st.spinner(t("Creating voice audio...")):
                                        treatment_audio = generate_tts_bytes(translated_treatment, lang_code=target_lang_code)
                                    if treatment_audio and len(treatment_audio) > 10:
                                        st.audio(treatment_audio, format="audio/mp3")
                                        st.download_button(label=t("⬇️ Download Audio Advice"), data=treatment_audio, file_name="treatment_advice.mp3", mime="audio/mpeg")
                        except Exception as e:
                            st.error(f"Oops, something went wrong: {e}")

    # --------------------------
    # TAB 2: Text Diagnosis (Bina Photo / Symptoms Only)
    # --------------------------
    with tab_text:
        st.markdown(f"#### {t('Describe the problem with your plant')}")
        st.caption(t('If you do not have a photo, simply type what is wrong (e.g. leaves are turning yellow, white powder on stem).'))
        
        symptoms_text = st.text_area(t("Type your plant symptoms here:"), height=200, placeholder=t("E.g., Tomato plant leaves are drying up from the edges and turning brown..."))
        
        if st.button(t("🔍 Analyze Symptoms Only"), use_container_width=True):
            if symptoms_text.strip():
                with st.spinner(t("Analyzing your description...")):
                    try:
                        advice_en = generate_treatment_from_text(symptoms_text, st.session_state['username'])
                        translated_advice = advice_en if selected_label == "English" else GoogleTranslator(source="auto", target=target_lang_code).translate(advice_en)
                        
                        st.subheader(t("AI Disease Analysis & Treatment 💡"))
                        st.info(translated_advice)
                        
                        with st.spinner(t("Creating voice audio...")):
                            symptom_audio = generate_tts_bytes(translated_advice, lang_code=target_lang_code)
                        if symptom_audio and len(symptom_audio) > 10:
                            st.audio(symptom_audio, format="audio/mp3")
                            st.download_button(label=t("⬇️ Download Audio Advice"), data=symptom_audio, file_name="symptom_advice.mp3", mime="audio/mpeg")
                            
                    except Exception as e:
                        st.error(f"Oops, something went wrong: {e}")
            else:
                st.warning(t("Please type some symptoms in the box first."))

    # --------------------------
    # Weather Risk Section
    # --------------------------
    st.markdown("---")
    
    # 🔥 FIX: Same robust glowing style applied to the Weather heading
    st.markdown(
        f"<h2 style='color: #10b981; text-shadow: 0 0 15px rgba(16, 185, 129, 0.4); font-weight: 800;'>{t('🌦️ Weather & Disease Risk')}</h2>", 
        unsafe_allow_html=True
    )
    
    if not st.session_state.get("weather_info"):
        st.info(t("Enter your city in the sidebar to see if the weather might harm your plants."))
    else:
        info = st.session_state["weather_info"]
        todays_items = get_todays_forecast_from_3h(info.get("forecast3h"))
        if todays_items:
            with st.spinner(t("Checking weather risks...")):
                raw_risk = assess_weather_risk_with_ai(todays_items, info.get("city",""), st.secrets.get("GEMINI_API_KEY"))
                translated_risk = raw_risk if selected_label == "English" else GoogleTranslator(source="auto", target=target_lang_code).translate(raw_risk)
            st.warning(translated_risk)
        else:
            st.info(t("Not enough weather data to calculate risk right now."))
            
    st.markdown("---")
    st.caption("© 2026 AI Plant Doctor — Your Smart Farming Buddy 🌾")
    
    # --------------------------
    # Floating Chat Widget
    # --------------------------
    chatbot_url = "https://light-yagami980.diaflow.app/public-chat/RGMNeOWpcT"
    floating_widget = f"""
    <style>
    #floatingChatContainer {{ position: fixed; bottom: 0; right: 0; z-index: 999999999; pointer-events: none; }}
    #chatButton {{ width: 65px; height: 65px; background: linear-gradient(135deg, #10b981, #059669); color: white; border-radius: 50%; border: none; font-size: 30px; cursor: pointer; margin: 20px; box-shadow: 0px 4px 15px rgba(16,185,129,0.5); pointer-events: auto; transition: transform 0.3s; }}
    #chatButton:hover {{ transform: scale(1.1); }}
    #chatFrame {{ width: 360px; height: 480px; border-radius: 14px; border: none; display: none; margin-right: 20px; margin-bottom: 95px; box-shadow: 0 10px 40px rgba(0,0,0,0.8); pointer-events: auto; }}
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
