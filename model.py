import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

data_dir = r"C:speech_recognition\processed_data"

emotions = ['neutral', 'happy', 'sad', 'angry']

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)

        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)

        features = np.hstack([
            np.mean(mfccs.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(mel.T, axis=0),
            np.mean(contrast.T, axis=0)
        ])

        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

print("Extracting features...")

X = []
y = []

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

print(f"Feature extraction complete. Total samples: {len(X)}, Feature shape: {X.shape[1]}")

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

model = Sequential()
model.add(Dense(256, input_shape=(X.shape[1],), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y_categorical.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Training the model...")

history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.2f}")

model.save("emotion_model.keras")  
np.save("label_classes.npy", le.classes_)

print("Model and labels saved!")
