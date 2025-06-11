# File: machine_learning/src/inference.py

import os
import numpy as np
import joblib
import cv2
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from machine_learning.src.config import CLASSES

# Dropbox model URLs
CNN_MODEL_URL = "https://www.dropbox.com/scl/fi/hycicdjgwyogidwpkmhin/cnn_model.h5?rlkey=6siy5ttso5cxzqb7xwetk8ve5&st=wvpmpeli&dl=1"
SVM_MODEL_URL = "https://www.dropbox.com/scl/fi/89ery357pf19i97tjl3c9/svm_model.pkl?rlkey=m64njjqrarqj7s1lp1wld9m9u&st=9n8m0ry1&dl=1"
RF_MODEL_URL  = "https://www.dropbox.com/scl/fi/rtvw1d9s8v432rip3v4dc/rf_model.pkl?rlkey=bt79zzljcvidel7ns7sxffngr&st=eqfqaye6&dl=1"

MODEL_DIR = "downloaded_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Lazy-loaded models
cnn_model = None
svm_model = None
rf_model = None
feature_model = None

def download_if_needed(url, filename):
    local_path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(local_path):
        print(f"🔽 Downloading {filename}...")
        r = requests.get(url)
        with open(local_path, "wb") as f:
            f.write(r.content)
    return local_path


def download_if_missing(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"Downloading {dest_path}...")
        r = requests.get(url)
        with open(dest_path, 'wb') as f:
            f.write(r.content)


def load_cnn():
    global cnn_model
    if cnn_model is None:
        path = download_if_needed(CNN_MODEL_URL, "cnn_model.h5")
        cnn_model = load_model(path)
    return cnn_model

def load_svm():
    global svm_model
    if svm_model is None:
        path = download_if_needed(SVM_MODEL_URL, "svm_model.pkl")
        svm_model = joblib.load(path)
    return svm_model

def load_rf():
    global rf_model
    if rf_model is None:
        path = download_if_needed(RF_MODEL_URL, "rf_model.pkl")
        rf_model = joblib.load(path)
    return rf_model

def load_feature_model():
    global feature_model
    if feature_model is None:
        vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        feature_model = Model(inputs=vgg.input, outputs=vgg.output)
    return feature_model

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    return img

def extract_features(img_array):
    model = load_feature_model()
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed = preprocess_input(img_array)
    features = model.predict(preprocessed)
    return features.reshape(1, -1)

def predict_class(image_path, model_type='cnn'):
    img_array = preprocess_image(image_path)

    if model_type == 'cnn':
        model = load_cnn()
        preds = model.predict(np.expand_dims(img_array, axis=0))
        return CLASSES[np.argmax(preds)]

    elif model_type == 'svm':
        model = load_svm()
        features = extract_features(img_array)
        return CLASSES[model.predict(features)[0]]

    elif model_type == 'rf':
        model = load_rf()
        features = extract_features(img_array)
        return CLASSES[model.predict(features)[0]]

    else:
        raise ValueError("Invalid model type. Choose from 'cnn', 'svm', 'rf'")
