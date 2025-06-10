# File: machine_learning/src/inference.py

import numpy as np
import joblib
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from machine_learning.src.config import CLASSES, MODEL_DIR

# Load models once
cnn_model = load_model(f"{MODEL_DIR}/cnn_model.h5")
svm_model = joblib.load(f"{MODEL_DIR}/svm_model.pkl")
rf_model = joblib.load(f"{MODEL_DIR}/rf_model.pkl")

# Load VGG16 for feature extraction
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
