# File: machine_learning/src/inference.py

import os
import requests
import numpy as np
import joblib
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from machine_learning.src.config import CLASSES, MODEL_DIR

# === DROPBOX DIRECT DOWNLOAD LINKS ===
CNN_MODEL_URL = "https://www.dropbox.com/scl/fi/hycicdjgwyogidwpkmhin/cnn_model.h5?rlkey=6siy5ttso5cxzqb7xwetk8ve5&st=wvpmpeli&dl=1"
SVM_MODEL_URL = "https://www.dropbox.com/scl/fi/89ery357pf19i97tjl3c9/svm_model.pkl?rlkey=m64njjqrarqj7s1lp1wld9m9u&st=9n8m0ry1&dl=1"
RF_MODEL_URL  = "https://www.dropbox.com/scl/fi/rtvw1d9s8v432rip3v4dc/rf_model.pkl?rlkey=bt79zzljcvidel7ns7sxffngr&st=eqfqaye6&dl=1"

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def download_if_missing(url, path):
    if not os.path.exists(path):
        print(f"[INFO] Downloading model: {os.path.basename(path)}")
        try:
            r = requests.get(url)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"[INFO] Successfully downloaded {os.path.basename(path)}")
        except Exception as e:
            raise RuntimeError(f"Failed to download {os.path.basename(path)}: {e}")

# === Download models if not already present ===
cnn_path = os.path.join(MODEL_DIR, "cnn_model.h5")
svm_path = os.path.join(MODEL_DIR, "svm_model.pkl")
rf_path  = os.path.join(MODEL_DIR, "rf_model.pkl")

download_if_missing(CNN_MODEL_URL, cnn_path)
download_if_missing(SVM_MODEL_URL, svm_path)
download_if_missing(RF_MODEL_URL, rf_path)

# === Load models ===
cnn_model = load_model(cnn_path)
svm_model = joblib.load(svm_path)
rf_model = joblib.load(rf_path)

# === VGG16 for feature extraction ===
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_model = Model(inputs=vgg_model.input, outputs=vgg_model.output)

def preprocess_image(image_path):
    """Loads and preprocesses the image from path."""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    return img

def extract_features(img_array):
    """Converts image to features using VGG16."""
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array)
    features = feature_model.predict(img_preprocessed)
    return features.reshape(1, -1)

def predict_class(image_path, model_type='cnn'):
    """Predict tumor class using the selected model."""
    img_array = preprocess_image(image_path)

    if model_type == 'cnn':
        input_tensor = np.expand_dims(img_array, axis=0)
        preds = cnn_model.predict(input_tensor)
        return CLASSES[np.argmax(preds)]

    elif model_type == 'svm':
        features = extract_features(img_array)
        preds = svm_model.predict(features)
        return CLASSES[preds[0]]

    elif model_type == 'rf':
        features = extract_features(img_array)
        preds = rf_model.predict(features)
        return CLASSES[preds[0]]

    else:
        raise ValueError("Invalid model type. Choose from 'cnn', 'svm', 'rf'")
