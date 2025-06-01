import sounddevice as sd
import numpy as np
import whisper
from transformers import pipeline
import tempfile
import os
import torch
import scipy.io.wavfile as wavfile

duration = 5  
sample_rate = 16000

print("🎙 Recording... Please speak clearly.")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
sd.wait()
print(" Recording done.\n")

with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
    wavfile.write(temp_audio_file.name, sample_rate, audio)
    audio_path = temp_audio_file.name

print(" Transcribing speech...")
model_whisper = whisper.load_model("base")  
result = model_whisper.transcribe(audio_path)
text = result['text'].strip()
print(f" Transcribed Text: {text}")

print(" Analyzing emotion...")
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)
emotion = classifier(text)[0]

print(f"\n Detected Emotion: {emotion['label']} (Confidence: {emotion['score']:.2f})")

os.remove(audio_path)
