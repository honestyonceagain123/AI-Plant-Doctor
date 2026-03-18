# app.py — AI Plant Doctor (clean, chatbot included, Option B: 3-hour forecast only, NO HUGGINGFACE/DIFFUSERS)
# Requirements: streamlit, torch, torchvision, gTTS, deep_translator, google-generativeai, requests, pandas, pillow, numpy, opencv-python

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
# Page config + device
# --------------------------
st.set_page_config(page_title="AI Plant Doctor", page_icon="🩺", layout="wide")

# 🔥 FIX FOR SIDEBAR: Force hide the Streamlit default multi-page navigation (like 'login', 'app')
st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Languages mapping
# --------------------------
LANGUAGES = {
    "English": "en",
    "Hindi (हिन्दी)": "hi",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
    "Bengali (বাংলা)": "bn",
    "Gujarati (ગુજરાતી)": "gu",
    "Punjabi (ਪੰਜਾਬੀ)": "pa",
    "Marathi (मराठी)": "mr",
    "Urdu (اردو)": "ur",
    "French (Français)": "fr",
    "German (Deutsch)": "de",
    "Spanish (Español)": "es",
    "Chinese (中文)": "zh-cn",
    "Japanese (日本語)": "ja"
}

# --------------------------
# Helper: device info banner
# --------------------------
def show_device_info():
    if DEVICE == "cuda":
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "CUDA GPU"
        st.sidebar.success(f"Using GPU: {name}")
    else:
        st.sidebar.warning("GPU not available — running on CPU.")

# --------------------------
# Model loaders (cached)
# --------------------------
@st.cache_resource
def load_plant_detector():
    """Loads ResNet18 binary classifier to check if image is actually a plant"""
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
        return None # Return None if not found, we'll skip the check gracefully

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
    if blur_score < 80:
        return False
    return True

def has_enough_green(image: Image.Image):
    img = np.array(image)
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    green_pixels = (g > r + 15) & (g > b + 15)
    green_ratio = np.sum(green_pixels) / green_pixels.size
    return green_ratio > 0.05

def is_plant_image(image, model):
    if model is None: 
        return True, 100.0 # Skip check if model missing
        
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
    if not isinstance(text, str):
        text = str(text)
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
    try:
        supported = gtts_langs.tts_langs()
    except Exception:
        supported = {"en":"English"}
    if lang_code not in supported:
        if lang_code.startswith("zh"):
            lang_code = "zh-cn" if "zh-cn" in supported else "zh"
        elif lang_code == "pa" and "pa" not in supported:
            lang_code = "hi" if "hi" in supported else "en"
        else:
            lang_code = "en"
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
        except Exception:
            return b""

# --------------------------
# Gemini AI treatment
# --------------------------
def generate_treatment_with_ai(disease_name: str) -> str:
    if "GEMINI_API_KEY" not in st.secrets:
        return "⚠️ GEMINI_API_KEY missing from Streamlit secrets."
    
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        You are an expert agronomist AI. A farmer's leaf is diagnosed as '{disease_name}'.
        Provide:
        1. A short cause summary
        2. Clear actionable treatment steps (step-by-step)
        3. Preventive tips for future
        Please keep sentences short and simple.
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ AI generation failed: {e}"

# --------------------------
# OpenWeather free helpers
# --------------------------
def geocode_city_to_coords(city: str, api_key: str):
    try:
        q = f"{city},IN"
        url = "http://api.openweathermap.org/geo/1.0/direct"
        params = {"q": q, "limit": 1, "appid": api_key}
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        return float(data[0]["lat"]), float(data[0]["lon"]), data[0].get("name", city)
    except Exception:
        return None

def get_current_weather(city: str, api_key: str):
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": f"{city},IN", "units": "metric", "appid": api_key}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def get_forecast_3h(city: str, api_key: str):
    """5-day / 3-hour forecast (free). We'll extract today's entries."""
    try:
        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {"q": f"{city},IN", "units": "metric", "appid": api_key}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def get_todays_forecast_from_3h(forecast_json):
    """Return list of forecast items (dicts) from forecast_json that belong to today (local date)."""
    if not forecast_json or "list" not in forecast_json:
        return []
    today = time.strftime("%Y-%m-%d", time.localtime())
    items = []
    for item in forecast_json["list"]:
        dt_txt = item.get("dt_txt", "")  # format "YYYY-MM-DD HH:MM:SS"
        if dt_txt.startswith(today):
            # keep relevant fields
            items.append({
                "time": dt_txt,
                "temp": item.get("main", {}).get("temp"),
                "feels_like": item.get("main", {}).get("feels_like"),
                "humidity": item.get("main", {}).get("humidity"),
                "desc": item.get("weather", [{}])[0].get("description", "")
            })
    return items

# --------------------------
# Weather -> risk function (Gemini Integration)
# --------------------------
def assess_weather_risk_with_ai(daily_forecasts: list, location_name: str = "", gemini_key: str = None):
    summary_lines = []
    for d in daily_forecasts:
        desc = d.get("description") or d.get("desc", "") or ""
        temp = d.get("temp", d.get("temp_day", None))
        humidity = d.get("humidity", None)
        summary_lines.append(f"{d.get('time', d.get('date',''))}: {desc}, temp {temp}°C, humidity {humidity}%")
    summary_text = "\n".join(summary_lines[:12])

    if gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            prompt = f"""
            You are an agricultural expert. Given the short-term weather summary for {location_name} below, provide:
            1) Short risk assessment for common plant diseases (fungal, bacterial, viral, pests) — Low/Moderate/High + 1-line reason each.
            2) 3 quick recommendations farmers can do today or this week.
            Weather:
            {summary_text}
            Keep it short and simple.
            """
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception:
            pass

    # Heuristic fallback
    fungal_score = 0
    pest_score = 0
    for d in daily_forecasts:
        h = d.get("humidity", 0) or 0
        desc = (d.get("desc", "") or d.get("description", "") or "").lower()
        if h >= 80 and ("rain" in desc or "shower" in desc or "thunder" in desc):
            fungal_score += 1
        if (d.get("temp", 0) or 0) >= 30 and h < 50:
            pest_score += 1

    fungal_risk = "High" if fungal_score >= 2 else ("Moderate" if fungal_score == 1 else "Low")
    pest_risk = "High" if pest_score >= 2 else ("Moderate" if pest_score == 1 else "Low")

    lines = []
    if fungal_risk != "Low":
        lines.append(f"Fungal risk {fungal_risk} due to humid/rainy periods today.")
    else:
        lines.append("Fungal risk Low based on today's forecast.")
    if pest_risk != "Low":
        lines.append(f"Pest risk {pest_risk} due to hot/dry windows today.")
    else:
        lines.append("Pest risk Low based on today's forecast.")

    recs = [
        "Remove standing water and improve drainage where possible.",
        "If feasible, consider preventive fungicide for vulnerable crops following local guidance.",
        "Increase scouting for pests and use traps or biological controls when found."
    ]

    return "Risk Summary:\n" + "\n".join(lines) + "\n\nRecommendations:\n" + "\n".join([f"- {r}" for r in recs])

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


# --------------------------
# Main UI
# --------------------------
def main():
    show_device_info()

    if "weather_info" not in st.session_state:
        st.session_state.weather_info = None
    if "manual_city" not in st.session_state:
        st.session_state.manual_city = ""

    st.sidebar.header("🌦️ Weather (Manual city — India)")
    st.sidebar.write("Country: **India 🇮🇳** (fixed)")
    st.session_state.manual_city = st.sidebar.text_input("Enter City (e.g., Mumbai)", value=st.session_state.manual_city)

    if st.sidebar.button("Get Weather (Manual)"):
        city = st.session_state.manual_city.strip()
        if city == "":
            st.sidebar.error("Please enter a city name.")
        elif "OPENWEATHER_KEY" not in st.secrets:
            st.sidebar.error("OPENWEATHER_KEY missing from .streamlit/secrets.toml")
        else:
            coords = geocode_city_to_coords(city, st.secrets["OPENWEATHER_KEY"])
            if not coords:
                cur = get_current_weather(city, st.secrets["OPENWEATHER_KEY"])
                if not cur or cur.get("cod") == 401:
                    if cur and cur.get("cod") == 401:
                        st.sidebar.error("OpenWeather API returned 401: Invalid API key. Check your OPENWEATHER_KEY in secrets.")
                    else:
                        st.sidebar.error("Could not resolve city or fetch weather. Try a major city spelling.")
                else:
                    name = cur.get("name", city)
                    forecast = get_forecast_3h(name, st.secrets["OPENWEATHER_KEY"])
                    st.session_state.weather_info = {"city": name, "lat": None, "lon": None, "current": cur, "forecast3h": forecast}
                    st.sidebar.success(f"Weather loaded for {name}")
            else:
                lat, lon, resolved_name = coords
                current = get_current_weather(resolved_name, st.secrets["OPENWEATHER_KEY"])
                forecast = get_forecast_3h(resolved_name, st.secrets["OPENWEATHER_KEY"])
                if not current:
                    st.sidebar.error("Failed to fetch current weather for resolved city.")
                else:
                    st.session_state.weather_info = {"city": resolved_name, "lat": lat, "lon": lon, "current": current, "forecast3h": forecast}
                    st.sidebar.success(f"Weather loaded for {resolved_name}")

    if st.session_state.get("weather_info"):
        info = st.session_state["weather_info"]
        cur = info.get("current", {})
        if cur:
            tval = cur.get("main", {}).get("temp"); hum = cur.get("main", {}).get("humidity")
            desc = cur.get("weather", [{}])[0].get("description","").title()
            st.sidebar.markdown(f"**{info.get('city','')}**")
            st.sidebar.write(f"🌡️ {tval} °C   💧 {hum}%")
            st.sidebar.write(desc)

    if "ui_lang_code" not in st.session_state:
        st.session_state.ui_lang_code = "en"
    selected_label = st.sidebar.selectbox("🌐 Translate & TTS language", list(LANGUAGES.keys()), index=list(LANGUAGES.keys()).index("English"))
    target_lang_code = LANGUAGES.get(selected_label, "en")
    st.session_state.ui_lang_code = target_lang_code
    translator_ui = GoogleTranslator(source="auto", target=target_lang_code)

    def t(text: str) -> str:
        try:
            return text if selected_label == "English" else translator_ui.translate(text)
        except Exception:
            return text

    st.title(t("🩺 AI Plant Doctor"))
    st.markdown(t("Upload a leaf photo to diagnose the disease and get AI treatment plus today's weather-aware risk."))

    uploaded_file = st.file_uploader(t("📤 Upload a leaf image"), type=["jpg", "jpeg", "png"])
    if not uploaded_file:
        st.caption(t("Tip: Use a clear close-up of the leaf."))
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption=t("Uploaded Image"), use_column_width=True)

    if st.button(t("🔍 Analyze")):
        
        # 1. Quality & Content checks first!
        if not is_clear_leaf_image(image):
            st.error(t("❌ Image is blurry or not a leaf"))
            return

        detector = load_plant_detector()
        with st.spinner(t("Checking if image is a plant...")):
            plant_ok, plant_conf = is_plant_image(image, detector)

        if not plant_ok or plant_conf < 90:
            st.error(t("❌ Uploaded image is NOT a clear plant"))
            st.info(t(f"Plant confidence: {plant_conf:.2f}%"))
            st.warning(t("Upload a close-up leaf with plain background"))
            return
            
        st.success(t(f"✅ Plant detected ({plant_conf:.2f}%)"))

        # 2. Main Classification
        try:
            model = load_classifier()
        except FileNotFoundError as e:
            st.error(str(e))
            return

        class_names = get_class_names()
        with st.spinner(t("Running classifier...")):
            try:
                prediction, confidence = predict(image, model, class_names)
            except Exception as e:
                st.error(f"Classifier failed: {e}")
                return

        st.success(f"🌿 {t('Prediction')}: {t(prediction)}")
        st.write(f"📊 {t('Model Confidence')}: {confidence:.2f}%")

        st.session_state.last_diagnosis = prediction

        # 3. Treatment
        if str(prediction).lower() == "healthy":
            st.success(t("Great news! The plant looks healthy. Keep up the good work!"))
        else:
            with st.spinner(t("Generating AI-based treatment...")):
                treatment_en = generate_treatment_with_ai(prediction)

            try:
                translated_treatment = treatment_en if selected_label == "English" else GoogleTranslator(source="auto", target=target_lang_code).translate(treatment_en)
            except Exception:
                translated_treatment = treatment_en

            st.session_state.last_treatment = treatment_en

            with st.expander(t("AI-Generated Treatment (original)")):
                st.write(treatment_en)

            st.subheader(t("AI-Generated Treatment"))
            st.write(translated_treatment)

            with st.spinner(t("Generating speech for treatment...")):
                treatment_audio = generate_tts_bytes(translated_treatment, lang_code=target_lang_code)
            if treatment_audio and len(treatment_audio) > 10:
                st.audio(treatment_audio, format="audio/mp3")
                st.download_button(label=t("⬇️ Download Treatment MP3"), data=treatment_audio, file_name="treatment.mp3", mime="audio/mpeg")
            else:
                st.error(t("TTS generation for treatment failed."))

        # 4. Weather & Forecast
        st.markdown("---")
        st.header(t("🌦️ Current Weather + Today's Forecast (3-hour slots)"))

        if not st.session_state.get("weather_info"):
            st.info(t("Set City in the sidebar and click 'Get Weather (Manual)' to fetch current weather and today's forecast (3-hour)."))
        else:
            info = st.session_state["weather_info"]
            cur = info.get("current")
            forecast3h = info.get("forecast3h")
            if cur:
                name = info.get("city", "")
                temp = cur.get("main", {}).get("temp")
                feels = cur.get("main", {}).get("feels_like")
                humidity = cur.get("main", {}).get("humidity")
                wind = cur.get("wind", {}).get("speed")
                desc = cur.get("weather", [{}])[0].get("description","").title()
                st.subheader(t(f"Current weather — {name}"))
                st.write(t(f"🌡️ Temperature: {temp}°C (Feels like {feels}°C)"))
                st.write(t(f"💧 Humidity: {humidity}%  ⚡ Wind: {wind} m/s"))
                st.write(t(f"📘 Condition: {desc}"))

            todays_items = get_todays_forecast_from_3h(forecast3h) if forecast3h else []
            if todays_items:
                st.subheader(t("Today's short-term forecast"))
                df_rows = []
                for it in todays_items:
                    df_rows.append({
                        "Time": it["time"].split(" ")[1][:5],
                        "Temp (°C)": it.get("temp"),
                        "Feels": it.get("feels_like"),
                        "Humidity (%)": it.get("humidity"),
                        "Condition": it.get("desc")
                    })
                st.table(df_rows[:6])
            else:
                st.info(t("No detailed 3-hour forecast available for today."))

            with st.spinner(t("Assessing weather-based disease risk...")):
                gemini_key = st.secrets.get("GEMINI_API_KEY")
                raw_risk = assess_weather_risk_with_ai(todays_items if todays_items else [{"desc": desc, "temp": temp, "humidity": humidity, "time": time.strftime("%Y-%m-%d %H:%M:%S")}], info.get("city",""), gemini_key)

            try:
                translated_risk = raw_risk if selected_label == "English" else GoogleTranslator(source="auto", target=target_lang_code).translate(raw_risk)
            except Exception:
                translated_risk = raw_risk

            st.subheader(t("Weather-based Disease Risk (today)"))
            st.write(translated_risk)

            with st.spinner(t("Generating speech for risk analysis...")):
                risk_audio = generate_tts_bytes(translated_risk, lang_code=target_lang_code)
            if risk_audio and len(risk_audio) > 10:
                st.audio(risk_audio, format="audio/mp3")
                st.download_button(label=t("⬇️ Download Risk MP3"), data=risk_audio, file_name="risk_analysis.mp3", mime="audio/mpeg")
            else:
                st.error(t("TTS generation for risk failed."))

    st.markdown("---")
    st.caption(t("© 2026 AI Plant Doctor — Smart Farming with Generative AI 🌾"))

# --------------------------
# Floating Chat Widget
# --------------------------
chatbot_url = "https://light-yagami980.diaflow.app/public-chat/RGMNeOWpcT"

floating_widget = f"""
<style>
#floatingChatContainer {{
    position: fixed;
    bottom: 0;
    right: 0;
    z-index: 999999999; /* Always on top */
    pointer-events: none; /* Prevent Streamlit capturing clicks */
}}

#chatButton {{
    width: 65px;
    height: 65px;
    background-color: #4CAF50;
    color: white;
    border-radius: 50%;
    border: none;
    font-size: 30px;
    cursor: pointer;
    margin: 20px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    pointer-events: auto; /* Button clickable */
}}

#chatFrame {{
    width: 360px;
    height: 480px;
    border-radius: 14px;
    border: none;
    display: none;
    margin-right: 20px;
    margin-bottom: 95px;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
    pointer-events: auto; /* Frame clickable */
}}
</style>

<div id="floatingChatContainer">
    <iframe id="chatFrame" src="{chatbot_url}"></iframe>
    <button id="chatButton">💬</button>
</div>

<script>
const btn = document.getElementById("chatButton");
const frame = document.getElementById("chatFrame");

btn.onclick = function() {{
    frame.style.display = frame.style.display === "none" ? "block" : "none";
}};
</script>
"""

components.html(floating_widget, height=0, width=0)

if __name__ == "__main__":
    main()
