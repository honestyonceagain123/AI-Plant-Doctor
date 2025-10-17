import streamlit as st
from deep_translator import GoogleTranslator
import speech_recognition as sr
from gtts import gTTS
import os
import tempfile
from pydub import AudioSegment
from pydub.playback import play

# Function to recognize speech from microphone
def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening... Speak now.")
        audio = r.listen(source, phrase_time_limit=5)
    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Speech not recognized."
    except sr.RequestError as e:
        return f"Could not request results; {e}"

# Function to play text as speech
def speak(text, lang_code):
    try:
        tts = gTTS(text=text, lang=lang_code)
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
            tts.save(fp.name)
            audio = AudioSegment.from_file(fp.name, format="mp3")
            play(audio)
    except Exception as e:
        st.error(f"❌ Error in speech playback: {e}")

# Language options
LANGS = {
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Odia": "or",
    "Urdu": "ur"
}

# Streamlit App
st.set_page_config(page_title="AiBhaskara — Voice Translator", page_icon="🎤")
st.title("🎙️ AiBhaskara — Voice-Powered Language Translator")
st.caption("Speak in English, and get spoken translation in your chosen language.")

# Language selection
target_lang = st.selectbox("🎯 Translate to:", list(LANGS.keys()))
lang_code = LANGS[target_lang]

# Record and translate
if st.button("🎧 Speak Now"):
    spoken_text = recognize_speech()
    st.write(f"🗣️ You said: {spoken_text}")

    if "not recognized" not in spoken_text.lower():
        translated = GoogleTranslator(source='auto', target=lang_code).translate(spoken_text)
        st.success(f"📝 Translated to {target_lang}: {translated}")

        if st.button("🔊 Speak Output"):
            speak(translated, lang_code)
