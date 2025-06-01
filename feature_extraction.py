import os
import numpy as np
import librosa

data_dir = r"C:speech_recognition\processed_data"

emotions = ['neutral', 'happy', 'sad', 'angry']

X = []
y = []

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error: {file_path}, {e}")
        return None

for emotion in emotions:
    folder = os.path.join(data_dir, emotion)
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        features = extract_features(file_path)
        if features is not None:
            X.append(features)
            y.append(emotion)

X = np.array(X)
y = np.array(y)

print(" Features extracted:")
print("X shape:", X.shape)
print("y shape:", y.shape)
