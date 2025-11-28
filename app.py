# app.py — AI Plant Doctor (clean, chatbot removed, Option B: 3-hour forecast only)
# Requirements: streamlit, torch, torchvision, diffusers, gTTS, deep_translator, openai, requests, pandas, pillow, numpy

import os
import re
import io
import time
import requests
import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from deep_translator import GoogleTranslator
from openai import OpenAI
# NOTE: Removed global diffusers import to prevent startup crashes. 
# They are now imported dynamically inside the generation functions.
from gtts import gTTS, lang as gtts_langs
import pandas as pd

# --------------------------
# Page config + device
# --------------------------
st.set_page_config(page_title="AI Plant Doctor", page_icon="🩺", layout="wide")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

IMG2IMG_RESIZE = (512, 512)
TEXT2IMG_RES = 512

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
# Helper: device info banner (will display once from main)
# --------------------------
def show_device_info():
    if DEVICE == "cuda":
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "CUDA GPU"
        st.sidebar.success(f"Using GPU: {name}")
    else:
        st.sidebar.warning("GPU not available — running on CPU (Slow for SD).")

# --------------------------
# Model loaders (cached)
# --------------------------
@st.cache_resource
def load_classifier(weight_path="best_plant_model.pth"):
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 240)
    if not os.path.exists(weight_path):
        st.error(f"Classifier weight missing: {weight_path}. Prediction disabled.")
        st.session_state.analysis_error = f"Classifier model file '{weight_path}' not found."
        return None
    try:
        state = torch.load(weight_path, map_location=DEVICE)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
            
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None
    return model

@st.cache_data
def get_class_names(train_dir="dataset/train"):
    # Fallback to generic names if directory missing (common in cloud deployment)
    if not os.path.isdir(train_dir):
        if "class_names" not in st.session_state:
             st.session_state.class_names = [f"Class_{i}" for i in range(240)]
        return st.session_state.class_names
    return sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

# --------------------------
# DEFERRED LOADING (Visuals)
# --------------------------
# No @st.cache_resource here! We load fresh or handle caching manually to avoid startup hangs.
def load_text2img(model_id="runwayml/stable-diffusion-v1-5"):
    # Dynamic import prevents ImportError on startup
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=DTYPE, safety_checker=None).to(DEVICE)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe

def load_img2img(model_id="runwayml/stable-diffusion-v1-5"):
    # Dynamic import prevents ImportError on startup
    from diffusers import StableDiffusionImg2ImgPipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=DTYPE, safety_checker=None).to(DEVICE)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe

# --------------------------
# Helpers & TTS cleaning
# --------------------------
def extract_species(disease_label):
    base = disease_label.split("__")[0] if "__" in disease_label else disease_label
    parts = [p for p in base.replace("___", "_").split("_") if p.isalpha()]
    return parts[0].lower() if parts else ''.join([c for c in disease_label if c.isalpha()])[:10].lower()

def clear_gpu():
    if DEVICE == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

def clean_text_for_tts(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`.+?`", " ", text)
    text = re.sub(r"[\-\*\+\>\#]", "", text) # Simplified regex
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def generate_tts_bytes(text: str, lang_code: str = "en") -> bytes:
    cleaned = clean_text_for_tts(text)
    if not cleaned: return b""
    try:
        supported = gtts_langs.tts_langs()
    except Exception:
        supported = {"en":"English"}
    
    if lang_code not in supported:
        if lang_code == "pa": lang_code = "hi" # Fallback for Punjabi -> Hindi
        else: lang_code = "en"
        
    audio_buf = io.BytesIO()
    try:
        tts = gTTS(text=cleaned, lang=lang_code)
        tts.write_to_fp(audio_buf)
        audio_buf.seek(0)
        return audio_buf.read()
    except Exception:
        return b""

# --------------------------
# OpenAI treatment (English)
# --------------------------
def generate_treatment_with_ai(disease_name: str) -> str:
    if "OPENAI_API_KEY" not in st.secrets:
        return "⚠️ OPENAI_API_KEY missing from Streamlit secrets."
    
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    prompt = f"""
You are an expert agronomist AI. A farmer's leaf is diagnosed as '{disease_name}'.
Provide:
1. A short cause summary
2. Clear actionable treatment steps (step-by-step)
3. Preventive tips for future
Please keep sentences short and simple.
"""
    try:
        response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":"You are an agricultural expert AI."},{"role":"user","content":prompt}],)
        if hasattr(response, "choices"):
            content = response.choices[0].message.content
        else:
            content = getattr(response, "text", str(response))
        return content.strip()
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
        if not data: return None
        return float(data[0]["lat"]), float(data[0]["lon"]), data[0].get("name", city)
    except Exception: return None

def get_current_weather(city: str, api_key: str):
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": f"{city},IN", "units": "metric", "appid": api_key}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception: return None

def get_forecast_3h(city: str, api_key: str):
    try:
        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {"q": f"{city},IN", "units": "metric", "appid": api_key}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception: return None

def get_todays_forecast_from_3h(forecast_json):
    if not forecast_json or "list" not in forecast_json: return []
    today = time.strftime("%Y-%m-%d", time.localtime())
    items = []
    for item in forecast_json["list"]:
        dt_txt = item.get("dt_txt", "")
        if dt_txt.startswith(today):
            items.append({
                "time": dt_txt,
                "temp": item.get("main", {}).get("temp"),
                "feels_like": item.get("main", {}).get("feels_like"),
                "humidity": item.get("main", {}).get("humidity"),
                "desc": item.get("weather", [{}])[0].get("description", "")
            })
    return items

def assess_weather_risk_with_ai(daily_forecasts: list, location_name: str = "", openai_key: str = None):
    summary_lines = []
    for d in daily_forecasts:
        desc = d.get("description") or d.get("desc", "") or ""
        temp = d.get("temp", d.get("temp_day", None))
        humidity = d.get("humidity", None)
        summary_lines.append(f"{d.get('time', d.get('date',''))}: {desc}, temp {temp}°C, humidity {humidity}%")
    summary_text = "\n".join(summary_lines[:12])

    if openai_key:
        try:
            client = OpenAI(api_key=openai_key)
            prompt = f"Given this weather for {location_name}:\n{summary_text}\nProvide short risk assessment (Fungal/Pest) and 3 quick tips for farmers."
            response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}],)
            if hasattr(response, "choices"): return response.choices[0].message.content.strip()
        except Exception: pass

    # Heuristic fallback
    return "Risk analysis fallback: Check local humidity and temperature for pest/fungal risks."

# --------------------------
# Prediction & Visuals
# --------------------------
def predict(image: Image.Image, model, class_names):
    if model is None: return "Model Error", 0.0
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, 1)
    idx = preds.item()
    # Safe index access
    name = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
    return name, float(conf.item() * 100.0)

def generate_healthy_leaf_img2img(base_pil_image: Image.Image, disease_label: str):
    pipe = load_img2img()
    species = extract_species(disease_label)
    prompt = (f"A realistic close-up photo of the same {species} leaf, perfectly healthy and vibrant green, same camera angle and lighting, photorealistic, no spots or lesions.")
    ref = base_pil_image.convert("RGB").resize(IMG2IMG_RESIZE)
    # CPU optimization: reduced steps
    gen = torch.Generator(device=DEVICE).manual_seed(42)
    out = pipe(prompt=prompt, image=ref, strength=0.55, num_inference_steps=20, guidance_scale=7.5, generator=gen)
    clear_gpu()
    return out.images[0]

def generate_healthy_product_text2img(disease_label: str):
    pipe = load_text2img()
    species = extract_species(disease_label)
    prompt = (f"A high-quality realistic image of a healthy {species} plant with ripe fruits, vibrant colors, natural outdoor lighting, detailed leaves and fruits.")
    # CPU optimization: reduced steps
    gen = torch.Generator(device=DEVICE).manual_seed(42)
    out = pipe(prompt=prompt, num_inference_steps=20, guidance_scale=8.5, generator=gen, height=TEXT2IMG_RES, width=TEXT2IMG_RES)
    clear_gpu()
    return out.images[0]

# --------------------------
# Main UI
# --------------------------
def main():
    # --- 1. Session State Setup ---
    if 'analyzed' not in st.session_state: st.session_state.analyzed = False
    if 'visuals_generated' not in st.session_state: st.session_state.visuals_generated = False
    if 'prediction' not in st.session_state: st.session_state.prediction = None
    if 'treatment_en' not in st.session_state: st.session_state.treatment_en = None
    if 'leaf_img' not in st.session_state: st.session_state.leaf_img = None
    if 'plant_img' not in st.session_state: st.session_state.plant_img = None
    
    show_device_info()

    # --- Weather Sidebar ---
    if "weather_info" not in st.session_state: st.session_state.weather_info = None
    if "manual_city" not in st.session_state: st.session_state.manual_city = ""

    st.sidebar.header("🌦️ Weather (Manual city — India)")
    st.sidebar.write("Country: **India 🇮🇳** (fixed)")
    st.session_state.manual_city = st.sidebar.text_input("Enter City (e.g., Mumbai)", value=st.session_state.manual_city)

    if st.sidebar.button("Get Weather (Manual)"):
        city = st.session_state.manual_city.strip()
        if city and "OPENWEATHER_KEY" in st.secrets:
            with st.spinner("Fetching weather..."):
                coords = geocode_city_to_coords(city, st.secrets["OPENWEATHER_KEY"])
                if coords:
                    lat, lon, rname = coords
                    cur = get_current_weather(rname, st.secrets["OPENWEATHER_KEY"])
                    forc = get_forecast_3h(rname, st.secrets["OPENWEATHER_KEY"])
                    if cur:
                        st.session_state.weather_info = {"city": rname, "current": cur, "forecast3h": forc}
                        st.sidebar.success(f"Loaded: {rname}")
                else:
                    st.sidebar.error("City not found.")
    
    if st.session_state.weather_info:
        info = st.session_state.weather_info
        cur = info.get("current", {})
        if cur:
            st.sidebar.write(f"🌡️ {cur.get('main',{}).get('temp')}°C")
            st.sidebar.write(cur.get('weather',[{}])[0].get('description','').title())

    # --- Main Content ---
    st.title("🩺 AI Plant Doctor")
    st.markdown("Upload a leaf photo to diagnose disease, get treatment, and see healthy projections.")

    uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])
    if not uploaded_file:
        st.caption("Tip: Use a clear close-up.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width='stretch')

    # --- Analysis Button ---
    if st.button("🔍 Analyze"):
        model = load_classifier()
        if model:
            class_names = get_class_names()
            with st.spinner("Analyzing leaf..."):
                pred, conf = predict(image, model, class_names)
                st.session_state.prediction = pred
                st.session_state.confidence = conf
                
                # Generate Treatment
                treatment = generate_treatment_with_ai(pred)
                st.session_state.treatment_en = treatment
                
                # Reset visuals state for new image
                st.session_state.visuals_generated = False
                st.session_state.leaf_img = None
                st.session_state.plant_img = None
                
                st.session_state.analyzed = True
                st.rerun()

    # --- Display Results (Persistent) ---
    if st.session_state.analyzed:
        st.success(f"🌿 Prediction: {st.session_state.prediction}")
        st.write(f"📊 Confidence: {st.session_state.confidence:.2f}%")
        
        st.subheader("AI Treatment Plan")
        st.write(st.session_state.treatment_en)
        
        # Audio Button (Will not erase results now!)
        tts_bytes = generate_tts_bytes(st.session_state.treatment_en)
        if tts_bytes:
            st.audio(tts_bytes, format="audio/mp3")
            st.download_button("⬇️ Download Audio", tts_bytes, "treatment.mp3", "audio/mpeg")

        st.markdown("---")
        
        # --- Visuals Generation Section ---
        st.subheader("Optional: AI Visuals (Healthy Projection)")
        
        # Show button if visuals haven't been generated yet
        if not st.session_state.visuals_generated:
            if st.button("🎨 Generate AI Visuals (Slow on CPU)"):
                with st.spinner("Generating visuals... This may take a few minutes on CPU."):
                    try:
                        leaf_img = generate_healthy_leaf_img2img(image, st.session_state.prediction)
                        plant_img = generate_healthy_product_text2img(st.session_state.prediction)
                        
                        st.session_state.leaf_img = leaf_img
                        st.session_state.plant_img = plant_img
                        st.session_state.visuals_generated = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Visual generation failed: {e}")
        
        # Show images if they exist
        if st.session_state.visuals_generated:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Healthy Leaf")
                st.image(st.session_state.leaf_img, width='stretch')
            with c2:
                st.subheader("Healthy Plant")
                st.image(st.session_state.plant_img, width='stretch')

    # Chatbot Widget (Always visible)
    import streamlit.components.v1 as components
    chatbot_html = """
    <div style="position: fixed; bottom: 0; right: 0; z-index: 9999;">
        <script>
        // Simple placeholder for chat widget script if needed
        </script>
        <button style="background:#4CAF50; color:white; border:none; border-radius:50%; width:60px; height:60px; font-size:30px; margin:20px; cursor:pointer; box-shadow:0px 4px 10px rgba(0,0,0,0.3);" onclick="alert('Chatbot would open here')">💬</button>
    </div>
    """
    components.html(chatbot_html, height=100)

if __name__ == "__main__":
    main()
