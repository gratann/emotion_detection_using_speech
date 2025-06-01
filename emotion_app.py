import streamlit as st
import sounddevice as sd
import numpy as np
import whisper
import tempfile
import os
from transformers import pipeline
import scipy.io.wavfile as wavfile

st.set_page_config(page_title=" Emotion Detector", layout="centered")

@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
    emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    return whisper_model, emotion_model

whisper_model, emotion_model = load_models()

st.title(" Speech Emotion Detection")
st.write("Click the button to record your voice, transcribe it, and detect the emotion.")

DURATION = 5
SAMPLE_RATE = 16000

if st.button("Start Recording"):
    st.info("Recording... Speak now!")
    

    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    st.success("Recording complete!")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        wavfile.write(tmp_file.name, SAMPLE_RATE, audio)
        audio_path = tmp_file.name


    st.write(" Transcribing speech...")
    result = whisper_model.transcribe(audio_path)
    transcript = result["text"].strip()
    st.text_area(" Transcribed Text", transcript, height=100)

    st.write(" Analyzing emotion...")
    emotion_result = emotion_model(transcript)[0]
    label = emotion_result['label']
    score = emotion_result['score']

    st.success(f" Detected Emotion: {label} (Confidence: {score:.2f})")

    os.remove(audio_path)
