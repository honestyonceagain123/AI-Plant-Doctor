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
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline # LINE 17 FIX: Ensuring the explicit import path is correct
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
    "Marathi (ਮਰਾਠੀ)": "mr",
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
    # NOTE: Assuming 240 classes based on previous logs/errors
    model.fc = nn.Linear(num_features, 240)
    if not os.path.exists(weight_path):
        # Fallback for deployment environments that might not have the model file in the root
        # This allows the app to load and show the UI, but prediction will fail.
        st.error(f"Classifier weight missing: {weight_path}. Prediction disabled.")
        st.session_state.analysis_error = f"Classifier model file '{weight_path}' not found in the repository."
        return None
    try:
        state = torch.load(weight_path, map_location=DEVICE)
        # Assuming the state dictionary is the whole file, if not, adjust key:
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
            
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        st.session_state.analysis_error = f"Error loading model weights: {e}"
        return None

    return model

@st.cache_data
def get_class_names():
    # Placeholder/Dummy list - MUST be replaced with your actual 240 names
    # This is currently the source of the 'class_175' problem.
    if "class_names" not in st.session_state or len(st.session_state.class_names) != 240:
        # Check if the model has a class_to_idx attached (rare in this flow)
        # Default to generic classes if the actual 240 names are unknown
        class_names = [f"Class_{i}" for i in range(240)]
        st.session_state.class_names = class_names
        return class_names
    return st.session_state.class_names


# --------------------------
# DEFERRED STABLE DIFFUSION LOADING FUNCTIONS
# These functions are called only when the user clicks the "Generate Visuals" button.
# They are not decorated with @st.cache_resource, but are called by the single cached 
# entry point, ensuring the download happens only once per session/machine.
# --------------------------
def _load_text2img_pipe(model_id="runwayml/stable-diffusion-v1-5"):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=DTYPE, safety_checker=None).to(DEVICE)
    if hasattr(pipe, "enable_attention_slicing"): pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try: pipe.enable_xformers_memory_efficient_attention()
        except Exception: pass
    if hasattr(pipe, "enable_vae_tiling"):
        try: pipe.enable_vae_tiling()
        except Exception: pass
    return pipe

def _load_img2img_pipe(model_id="runwayml/stable-diffusion-v1-5"):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=DTYPE, safety_checker=None).to(DEVICE)
    if hasattr(pipe, "enable_attention_slicing"): pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try: pipe.enable_xformers_memory_efficient_attention()
        except Exception: pass
    if hasattr(pipe, "enable_vae_tiling"):
        try: pipe.enable_vae_tiling()
        except Exception: pass
    return pipe

@st.cache_resource
def load_visuals_models(pipe_type: str):
    """
    Cached wrapper to ensure the massive models are only downloaded once per session/machine
    when requested by the user's click.
    """
    try:
        if pipe_type == 'text2img':
            return _load_text2img_pipe()
        elif pipe_type == 'img2img':
            return _load_img2img_pipe()
        return None
    except Exception as e:
        st.session_state.visuals_error = f"AI Visuals Model Load Error: {e}"
        return None


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
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        except Exception:
            pass

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
    if not cleaned: return b""
    try:
        # Check for language support
        supported = gtts_langs.tts_langs()
    except Exception:
        supported = {"en":"English"}
        
    if lang_code not in supported:
        # Fallback for unsupported codes
        if lang_code.startswith("zh"): lang_code = "zh-cn"
        elif lang_code == "pa": lang_code = "hi"
        else: lang_code = "en"
    
    audio_buf = io.BytesIO()
    try:
        tts = gTTS(text=cleaned, lang=lang_code)
        tts.write_to_fp(audio_buf)
        audio_buf.seek(0)
        return audio_buf.read()
    except Exception:
        # Fallback to English if target language fails
        try:
            tts = gTTS(text=cleaned, lang="en")
            audio_buf = io.BytesIO()
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
    
    # Check if key is valid (sometimes key is present but invalid)
    if not st.secrets["OPENAI_API_KEY"] or st.secrets["OPENAI_API_KEY"].startswith("sk-YOUR"):
         return "⚠️ OPENAI_API_KEY is present but appears to be a placeholder or invalid."

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
        content = response.choices[0].message.content
        return content.strip()
    except Exception as e:
        return f"⚠️ AI generation failed: {e}"

# --------------------------
# OpenWeather free helpers: geocode, current, forecast(3h)
# --------------------------
# (Weather helper functions remain the same for brevity)
# (Assuming geocode_city_to_coords, get_current_weather, get_forecast_3h, etc. are correct)
# --- [REDACTED WEATHER HELPER FUNCTIONS] ---
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

def assess_weather_risk_with_ai(daily_forecasts: list, location_name: str = "", openai_key: str = None):
    # daily_forecasts is a list of dicts with keys like 'time'/'temp'/'humidity'/'description'
    summary_lines = []
    for d in daily_forecasts:
        desc = d.get("description") or d.get("desc", "") or ""
        temp = d.get("temp", d.get("temp_day", None))
        humidity = d.get("humidity", None)
        summary_lines.append(f"{d.get('time', d.get('date',''))}: {desc}, temp {temp}°C, humidity {humidity}%")
    summary_text = "\n".join(summary_lines[:12])  # use up to several slots for context

    # Try OpenAI if key present
    if openai_key and not openai_key.startswith("sk-YOUR"):
        try:
            client = OpenAI(api_key=openai_key)
            prompt = f"""
You are an agricultural expert. Given the short-term weather summary for {location_name} below, provide:
1) Short risk assessment for common plant diseases (fungal, bacterial, viral, pests) — Low/Moderate/High + 1-line reason each.
2) 3 quick recommendations farmers can do today or this week.
Weather:
{summary_text}
Keep it short and simple.
"""
            response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}],)
            return response.choices[0].message.content.strip()
        except Exception:
            pass # fall through to heuristic

    # Heuristic fallback (simple)
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
    if fungal_risk != "Low": lines.append(f"Fungal risk {fungal_risk} due to humid/rainy periods today.")
    else: lines.append("Fungal risk Low based on today's forecast.")
    if pest_risk != "Low": lines.append(f"Pest risk {pest_risk} due to hot/dry windows today.")
    else: lines.append("Pest risk Low based on today's forecast.")

    recs = [
        "Remove standing water and improve drainage where possible.",
        "If feasible, consider preventive fungicide for vulnerable crops following local guidance.",
        "Increase scouting for pests and use traps or biological controls when found."
    ]

    return "Risk Summary:\n" + "\n".join(lines) + "\n\nRecommendations:\n" + "\n".join([f"- {r}" for r in recs])


# --------------------------
# Prediction & SD helpers
# --------------------------
def predict(image: Image.Image, model, class_names):
    if model is None: return "Classifier Model Unavailable", 0.0
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, 1)
    idx = preds.item()
    name = class_names[idx] if class_names and idx < len(class_names) else f"class_{idx}"
    return name, float(conf.item() * 100.0)

def generate_healthy_leaf_img2img(base_pil_image: Image.Image, disease_label: str, strength=0.55, steps=25, guidance_scale=7.5):
    pipe = load_visuals_models('img2img')
    if pipe is None: raise Exception(st.session_state.get('visuals_error', "Model pipeline failed."))
    
    species = extract_species(disease_label)
    prompt = (f"A realistic close-up photo of the same {species} leaf, perfectly healthy and vibrant green, same camera angle and lighting, photorealistic, no spots or lesions.")
    ref = base_pil_image.convert("RGB").resize(IMG2IMG_RESIZE)
    gen = torch.Generator(device=DEVICE).manual_seed(np.random.randint(0, 2**31 - 1))
    out = pipe(prompt=prompt, image=ref, strength=strength, num_inference_steps=int(steps), guidance_scale=float(guidance_scale), generator=gen)
    clear_gpu()
    return out.images[0]

def generate_healthy_product_text2img(disease_label: str, steps=30, guidance_scale=8.5):
    pipe = load_visuals_models('text2img')
    if pipe is None: raise Exception(st.session_state.get('visuals_error', "Model pipeline failed."))

    species = extract_species(disease_label)
    prompt = (f"A high-quality realistic image of a healthy {species} plant with ripe fruits, vibrant colors, natural outdoor lighting, detailed leaves and fruits.")
    gen = torch.Generator(device=DEVICE).manual_seed(np.random.randint(0, 2**31 - 1))
    out = pipe(prompt=prompt, num_inference_steps=int(steps), guidance_scale=float(guidance_scale), generator=gen, height=TEXT2IMG_RES, width=TEXT2IMG_RES)
    clear_gpu()
    return out.images[0]

# --------------------------
# Main UI (sidebar moved inside main; chatbot removed)
# --------------------------
def main():
    # --- SESSION STATE INITIALIZATION ---
    if 'analyzed' not in st.session_state: st.session_state.analyzed = False
    if 'prediction' not in st.session_state: st.session_state.prediction = ""
    if 'confidence' not in st.session_state: st.session_state.confidence = 0.0
    if 'treatment_en' not in st.session_state: st.session_state.treatment_en = ""
    if 'translated_treatment' not in st.session_state: st.session_state.translated_treatment = ""
    if 'translated_risk' not in st.session_state: st.session_state.translated_risk = ""
    if 'leaf_img' not in st.session_state: st.session_state.leaf_img = None
    if 'plant_img' not in st.session_state: st.session_state.plant_img = None
    if 'uploaded_image' not in st.session_state: st.session_state.uploaded_image = None
    if 'treatment_audio_bytes' not in st.session_state: st.session_state.treatment_audio_bytes = b""
    if 'risk_audio_bytes' not in st.session_state: st.session_state.risk_audio_bytes = b""
    if 'risk_analyzed' not in st.session_state: st.session_state.risk_analyzed = False
    if 'visuals_generated' not in st.session_state: st.session_state.visuals_generated = False
    if 'visuals_error' not in st.session_state: st.session_state.visuals_error = None
    if 'analysis_error' not in st.session_state: st.session_state.analysis_error = ""


    # show GPU info once in sidebar
    show_device_info()

    # Sidebar: weather manual input (India only)
    if "weather_info" not in st.session_state:
        st.session_state.weather_info = None
    if "manual_city" not in st.session_state:
        st.session_state.manual_city = ""

    st.sidebar.header("🌦️ Weather (Manual city — India)")
    st.sidebar.write("Country: **India 🇮🇳** (fixed)")
    
    # ----------------------------------------------------
    # Weather Input Logic
    # ----------------------------------------------------
    city_input = st.sidebar.text_input("Enter City (e.g., Mumbai)", value=st.session_state.manual_city, key="city_input_key")

    def fetch_weather():
        city = city_input.strip()
        st.session_state.manual_city = city # Update session state for persistence

        if city == "":
            st.sidebar.error("Please enter a city name.")
            return
        if "OPENWEATHER_KEY" not in st.secrets or not st.secrets["OPENWEATHER_KEY"]:
            st.sidebar.error("OPENWEATHER_KEY missing from Streamlit secrets.")
            return
        
        api_key = st.secrets["OPENWEATHER_KEY"]

        with st.spinner("Fetching weather data..."):
            coords = geocode_city_to_coords(city, api_key)
            if not coords:
                st.sidebar.error("Could not resolve city or fetch weather. Try a major city spelling.")
                return

            lat, lon, resolved_name = coords
            current = get_current_weather(resolved_name, api_key)
            forecast = get_forecast_3h(resolved_name, api_key)

            if not current:
                st.sidebar.error("Failed to fetch current weather.")
                return

            st.session_state.weather_info = {
                "city": resolved_name, "lat": lat, "lon": lon, "current": current, "forecast3h": forecast
            }
            # Reset risk state when new weather data is fetched
            st.session_state.risk_analyzed = False
            st.sidebar.success(f"Weather loaded for {resolved_name}")

    st.sidebar.button("Get Weather (Manual)", on_click=fetch_weather)

    # Sidebar quick summary
    if st.session_state.get("weather_info"):
        info = st.session_state["weather_info"]
        cur = info.get("current", {})
        if cur:
            tval = cur.get("main", {}).get("temp"); hum = cur.get("main", {}).get("humidity")
            desc = cur.get("weather", [{}])[0].get("description","").title()
            st.sidebar.markdown(f"**{info.get('city','')}**")
            st.sidebar.write(f"🌡️ {tval} °C   💧 {hum}%")
            st.sidebar.write(desc)

    # Main content
    # language selection (UI & TTS)
    if "ui_lang_code" not in st.session_state: st.session_state.ui_lang_code = "en"
    
    selected_label = st.sidebar.selectbox("🌐 Translate & TTS language", 
                                        list(LANGUAGES.keys()), 
                                        index=list(LANGUAGES.keys()).index("English"))
    
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
    
    # Store uploaded file persistently only if a new one is uploaded
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.session_state.uploaded_image = image
            st.session_state.analyzed = False # Reset analysis flag if new image
            st.session_state.visuals_generated = False # Reset visuals flag
            st.session_state.leaf_img = None
            st.session_state.plant_img = None
        except Exception as e:
            st.error(f"Error loading image: {e}")
            st.session_state.uploaded_image = None
            return
    elif st.session_state.uploaded_image:
        image = st.session_state.uploaded_image
    else:
        st.caption(t("Tip: Use a clear close-up of the leaf."))
        return


    # Display the image persistently
    st.image(image, caption=t("Uploaded Image"), use_container_width=True)

    # ----------------------------------------------------
    # ANALYSIS BUTTON LOGIC
    # ----------------------------------------------------
    if st.button(t("🔍 Analyze")):
        st.session_state.analyzed = False # Ensure we don't display old results if analysis fails
        st.session_state.visuals_generated = False # Reset visuals
        st.session_state.treatment_audio_bytes = b"" # Clear audio cache
        st.session_state.risk_audio_bytes = b"" # Clear audio cache
        st.session_state.leaf_img = None
        st.session_state.plant_img = None
        st.session_state.analysis_error = "" # Clear previous error
        
        # 1. Classification
        model = load_classifier()
        if model is None: 
            # The error message was already set in load_classifier via st.session_state.analysis_error
            st.rerun() 
            return

        class_names = get_class_names()
        with st.spinner(t("Running classifier...")):
            prediction, confidence = predict(image, model, class_names)
            st.session_state.prediction = prediction
            st.session_state.confidence = confidence

        # 2. Treatment Generation
        with st.spinner(t("Generating AI-based treatment...")):
            treatment_en = generate_treatment_with_ai(prediction)
            
            # Translate treatment
            try:
                translated_treatment = treatment_en if selected_label == "English" else GoogleTranslator(source="auto", target=target_lang_code).translate(treatment_en)
            except Exception:
                translated_treatment = treatment_en
            
            st.session_state.treatment_en = treatment_en
            st.session_state.translated_treatment = translated_treatment
            
        # 3. Finalize Analysis (Set flag to true after all data is saved)
        st.session_state.analyzed = True
        st.rerun() # Rerun to display persistent results

    # ----------------------------------------------------
    # PERSISTENT RESULTS DISPLAY
    # ----------------------------------------------------
    if st.session_state.analysis_error:
        st.error(st.session_state.analysis_error)
    
    if st.session_state.analyzed:
        # Display Prediction
        st.success(f"🌿 {t('Prediction')}: {t(st.session_state.prediction)}")
        st.write(f"📊 {t('Model Confidence')}: {st.session_state.confidence:.2f}%")

        # Display Treatment
        with st.expander(t("AI-Generated Treatment (original)")):
            st.write(st.session_state.treatment_en)

        st.subheader(t("AI-Generated Treatment"))
        st.write(st.session_state.translated_treatment)
        
        # Treatment TTS & Download Button
        # We only generate audio if the state is empty (prevents regenerating on every click)
        if not st.session_state.treatment_audio_bytes or st.session_state.get('last_tts_text') != st.session_state.translated_treatment:
             st.session_state.treatment_audio_bytes = generate_tts_bytes(st.session_state.translated_treatment, lang_code=target_lang_code)
             st.session_state.last_tts_text = st.session_state.translated_treatment
             
        if st.session_state.treatment_audio_bytes and len(st.session_state.treatment_audio_bytes) > 10:
            st.audio(st.session_state.treatment_audio_bytes, format="audio/mp3")
            st.download_button(
                label=t("⬇️ Download Treatment MP3"), 
                data=st.session_state.treatment_audio_bytes, 
                file_name="treatment.mp3", 
                mime="audio/mpeg"
            )
        else:
            st.error(t("TTS generation for treatment failed."))


        # ----------------------------------------------------
        # WEATHER AND RISK SECTION
        # ----------------------------------------------------
        st.markdown("---")
        st.header(t("🌦️ Current Weather + Today's Forecast (3-hour slots)"))

        if not st.session_state.get("weather_info"):
            st.info(t("Set City in the sidebar and click 'Get Weather (Manual)' to fetch current weather and today's forecast (3-hour)."))
        else:
            # Run risk analysis only if weather was just updated or if we haven't analyzed yet
            if not st.session_state.risk_analyzed:
                with st.spinner(t("Assessing weather-based disease risk...")):
                    info = st.session_state["weather_info"]
                    todays_items = get_todays_forecast_from_3h(info.get("forecast3h"))
                    
                    openai_key = st.secrets.get("OPENAI_API_KEY")
                    raw_risk = assess_weather_risk_with_ai(todays_items, info.get("city",""), openai_key)
                    
                    try:
                        translated_risk = raw_risk if selected_label == "English" else GoogleTranslator(source="auto", target=target_lang_code).translate(raw_risk)
                    except Exception:
                        translated_risk = raw_risk
                    
                    st.session_state.translated_risk = translated_risk
                    st.session_state.risk_analyzed = True
                    st.session_state.risk_audio_bytes = b"" # Reset risk audio
                    st.rerun()
            
            # Display Weather and Risk Details (Reading from session state)
            info = st.session_state["weather_info"]
            cur = info.get("current")
            if cur:
                name = info.get("city", "")
                temp = cur.get("main", {}).get("temp"); feels = cur.get("main", {}).get("feels_like")
                humidity = cur.get("main", {}).get("humidity"); wind = cur.get("wind", {}).get("speed")
                desc = cur.get("weather", [{}])[0].get("description","").title()
                st.subheader(t(f"Current weather — {name}"))
                st.write(t(f"🌡️ Temperature: {temp}°C (Feels like {feels}°C)"))
                st.write(t(f"💧 Humidity: {humidity}%  ⚡ Wind: {wind} m/s"))
                st.write(t(f"📘 Condition: {desc}"))

            todays_items = get_todays_forecast_from_3h(info.get("forecast3h"))
            if todays_items:
                st.subheader(t("Today's short-term forecast"))
                df_rows = [{"Time": it["time"].split(" ")[1][:5], "Temp (°C)": it.get("temp"), "Feels": it.get("feels_like"), "Humidity (%)": it.get("humidity"), "Condition": it.get("desc")} for it in todays_items]
                st.table(df_rows[:6])
            else:
                st.info(t("No detailed 3-hour forecast available for today."))

            # Display Weather Risk
            st.subheader(t("Weather-based Disease Risk (today)"))
            st.write(st.session_state.translated_risk)
            
            # Risk TTS & Download Button
            if not st.session_state.risk_audio_bytes:
                 st.session_state.risk_audio_bytes = generate_tts_bytes(st.session_state.translated_risk, lang_code=target_lang_code)

            if st.session_state.risk_audio_bytes and len(st.session_state.risk_audio_bytes) > 10:
                st.audio(st.session_state.risk_audio_bytes, format="audio/mp3")
                st.download_button(
                    label=t("⬇️ Download Risk MP3"), 
                    data=st.session_state.risk_audio_bytes, 
                    file_name="risk_analysis.mp3", 
                    mime="audio/mpeg"
                )
            else:
                st.error(t("TTS generation for risk failed."))


        # ----------------------------------------------------
        # OPTIONAL AI VISUALS (Moved behind a dedicated button)
        # ----------------------------------------------------
        st.markdown("---")
        st.header(t("Optional: AI visuals (healthy leaf & plant)"))
        
        # New Button Logic for image generation
        if st.session_state.prediction:
            if not st.session_state.visuals_generated:
                if st.button(t("🎨 Generate AI Visuals (Slow)")):
                    if "HUGGINGFACE_TOKEN" not in st.secrets or not st.secrets["HUGGINGFACE_TOKEN"]:
                        st.error(t("HUGGINGFACE_TOKEN missing from secrets. Cannot generate images."))
                    else:
                        with st.spinner(t("Generating AI visuals... (This may take several minutes on cloud CPU)") + "..."):
                            try:
                                leaf_img = generate_healthy_leaf_img2img(image, st.session_state.prediction)
                                plant_img = generate_healthy_product_text2img(st.session_state.prediction)
                                st.session_state.leaf_img = leaf_img
                                st.session_state.plant_img = plant_img
                                st.session_state.visuals_generated = True
                                st.rerun() # Rerun to show images
                            except Exception as e:
                                st.error(f"AI visuals failed: {e}")
                                clear_gpu()
            
            if st.session_state.visuals_generated:
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader(t("Healthy Leaf (AI-Repaired)"))
                    st.image(st.session_state.leaf_img, use_column_width=True)
                with c2:
                    st.subheader(t("Healthy Plant (AI)"))
                    st.image(st.session_state.plant_img, use_container_width=True) # Use container width for better look
            elif not st.session_state.visuals_generated and st.session_state.analyzed:
                st.info(t("Click the 'Generate AI Visuals' button above to create model images."))


    st.markdown("---")
    st.caption(t("© 2025 AI Plant Doctor — Smart Farming with Generative AI 🌾"))

# --------------------------
# Floating Chat Widget (retained from original file)
# --------------------------
import streamlit.components.v1 as components
chatbot_url = "https://light-yagami980.diaflow.app/public-chat/RGMNeOWpcT"
floating_widget = f"""
<style>
#floatingChatContainer {{
    position: fixed;
    bottom: 0;
    right: 0;
    z-index: 999999999;
    pointer-events: none;
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
    pointer-events: auto;
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
    pointer-events: auto;
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
