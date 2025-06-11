import numpy as np
import joblib
import cv2
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.models import Model
from machine_learning.src.config import CLASSES, MODEL_DIR, CNN_MODEL_URL, SVM_MODEL_URL, RF_MODEL_URL

# Lazy-loaded models (set to None at startup)
cnn_model = None
svm_model = None
rf_model = None
feature_model = None

def download_model(url, path):
    if not os.path.exists(path):
        print(f"Downloading model from {url} to {path}")
        r = requests.get(url)
        with open(path, 'wb') as f:
            f.write(r.content)

def load_models():
    global cnn_model, svm_model, rf_model, feature_model

    # Download if missing
    download_model(CNN_MODEL_URL, f"{MODEL_DIR}/cnn_model.h5")
    download_model(SVM_MODEL_URL, f"{MODEL_DIR}/svm_model.pkl")
    download_model(RF_MODEL_URL, f"{MODEL_DIR}/rf_model.pkl")

    # Load models
    if cnn_model is None:
        cnn_model = load_model(f"{MODEL_DIR}/cnn_model.h5")

    if svm_model is None:
        svm_model = joblib.load(f"{MODEL_DIR}/svm_model.pkl")

    if rf_model is None:
        rf_model = joblib.load(f"{MODEL_DIR}/rf_model.pkl")

    if feature_model is None:
        vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        feature_model = Model(inputs=vgg.input, outputs=vgg.output)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    return img

def extract_features(img_array):
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array)
    features = feature_model.predict(img_preprocessed)
    return features.reshape(1, -1)

def predict_class(image_path, model_type='cnn'):
    load_models()  # Load models only when needed
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
        raise ValueError("Invalid model type. Choose from 'cnn', 'svm', or 'rf'")
