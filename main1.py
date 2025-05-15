from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
import librosa
import numpy as np
import pickle
from tensorflow.keras.models import model_from_json
app = FastAPI(title="Speech Emotion Recognition API")

# Configuration constants
MODEL_JSON_PATH = 'CNN_model.json'
MODEL_WEIGHTS_PATH = 'CNN_model.weights.h5'
SCALER_PATH = 'scaler2.pickle'
ENCODER_PATH = 'encoder2.pickle'
TARGET_FEATURE_SIZE = 1620  # must match scaler expectation

# 1) Load model, scaler, encoder
with open(MODEL_JSON_PATH, 'r') as f:
    loaded_model = model_from_json(f.read())
loaded_model.load_weights(MODEL_WEIGHTS_PATH)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
with open(ENCODER_PATH, 'rb') as f:
    encoder = pickle.load(f)

# Feature extraction helpers

def zcr(data, frame_length=2048, hop_length=512):
    return np.squeeze(
        librosa.feature.zero_crossing_rate(
            y=data, frame_length=frame_length, hop_length=hop_length
        )
    )

def rmse(data, frame_length=2048, hop_length=512):
    return np.squeeze(
        librosa.feature.rms(
            y=data, frame_length=frame_length, hop_length=hop_length
        )
    )

def mfcc(data, sr, n_fft=2048, hop_length=512):
    mfccs = librosa.feature.mfcc(
        y=data, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length
    )
    return np.ravel(mfccs.T)


def extract_features(data, sr, frame_length=2048, hop_length=512):
    return np.hstack((
        zcr(data, frame_length, hop_length),
        rmse(data, frame_length, hop_length),
        mfcc(data, sr, frame_length, hop_length)
    ))


def predict_from_raw(data, sr):
    # Extract and resize
    features = extract_features(data, sr)
    if len(features) < TARGET_FEATURE_SIZE:
        features = np.pad(features, (0, TARGET_FEATURE_SIZE - len(features)))
    else:
        features = features[:TARGET_FEATURE_SIZE]

    # Scale and reshape
    features = features.reshape(1, -1)
    features_scaled = scaler.transform(features)
    features_scaled = np.expand_dims(features_scaled, axis=2)

    # Predict
    preds = loaded_model.predict(features_scaled)
    label = encoder.inverse_transform(preds)
    return label[0][0]


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    tmp_path = f"temp_{file.filename}"
    try:
        # Save upload to a temp file
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load raw audio preserving original sampling rate
        data, sr = librosa.load(tmp_path, sr=None)
        emotion = predict_from_raw(data, sr)

        return JSONResponse(content={"emotion": emotion})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# To run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
