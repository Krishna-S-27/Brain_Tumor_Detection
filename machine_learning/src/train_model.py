import os
import time
import joblib
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from src.config import PROCESSED_DIR, MODEL_DIR, IMG_HEIGHT, IMG_WIDTH, CLASSES

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# ========== Load Preprocessed Data ==========
def load_data():
    try:
        X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
        X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
        y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
        y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
        print(f"[INFO] Loaded {len(X_train)} training and {len(X_test)} test samples.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return None, None, None, None

# ========== Resize for VGG16 ==========
def resize_images(X, size=(224, 224)):
    return np.array([cv2.resize(img, size) for img in X])

# ========== VGG16 Feature Extractor ==========
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_model = Model(inputs=vgg_model.input, outputs=vgg_model.output)

def extract_features(X):
    features = feature_model.predict(X, batch_size=32, verbose=1)
    return features.reshape(features.shape[0], -1)

# ========== SVM Training ==========
def train_svm(X_train, y_train):
    model = SVC(kernel='linear', probability=True)
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    joblib.dump(model, os.path.join(MODEL_DIR, 'svm_model.pkl'))
    return model, end - start

# ========== Random Forest Training ==========
def train_rf(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100)
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    joblib.dump(model, os.path.join(MODEL_DIR, 'rf_model.pkl'))
    return model, end - start

# ========== CNN Training ==========
def train_cnn(X_train, y_train, X_val, y_val):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(CLASSES), activation='softmax')
    ])
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    start = time.time()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=2)
    end = time.time()
    model.save(os.path.join(MODEL_DIR, 'cnn_model.h5'))
    return model, end - start

# ========== Main Pipeline ==========
def main():
    print("[INFO] Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    if X_train is None:
        return

    # Convert one-hot to labels for classical models
    y_train_labels = np.argmax(y_train, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    # VGG16 Feature Preparation
    print("[INFO] Preparing VGG16 features for classical models...")
    X_train_resized = preprocess_input(resize_images(X_train, size=(224, 224)))
    X_test_resized = preprocess_input(resize_images(X_test, size=(224, 224)))
    X_train_features = extract_features(X_train_resized)
    X_test_features = extract_features(X_test_resized)

    # ========== SVM ==========
    print("[INFO] Training SVM...")
    svm_model, svm_time = train_svm(X_train_features, y_train_labels)
    svm_preds = svm_model.predict(X_test_features)
    svm_acc = accuracy_score(y_test_labels, svm_preds)
    print(f"[RESULT] SVM Accuracy: {svm_acc:.4f} | Training Time: {svm_time:.2f}s")

    # ========== Random Forest ==========
    print("[INFO] Training Random Forest...")
    rf_model, rf_time = train_rf(X_train_features, y_train_labels)
    rf_preds = rf_model.predict(X_test_features)
    rf_acc = accuracy_score(y_test_labels, rf_preds)
    print(f"[RESULT] RF Accuracy: {rf_acc:.4f} | Training Time: {rf_time:.2f}s")

    # ========== CNN ==========
    print("[INFO] Training CNN...")
    cnn_model, cnn_time = train_cnn(X_train, y_train, X_test, y_test)
    cnn_score = cnn_model.evaluate(X_test, y_test, verbose=0)
    print(f"[RESULT] CNN Accuracy: {cnn_score[1]:.4f} | Training Time: {cnn_time:.2f}s")


def train_model():
    """
    Runs the training pipeline and returns a dictionary of metrics.
    """
    X_train, X_test, y_train, y_test = load_data()
    if X_train is None:
        return {"accuracy": 0.0}

    # Convert one-hot to labels for classical models
    y_train_labels = np.argmax(y_train, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    # VGG16 Feature Preparation
    X_train_resized = preprocess_input(resize_images(X_train, size=(224, 224)))
    X_test_resized = preprocess_input(resize_images(X_test, size=(224, 224)))
    X_train_features = extract_features(X_train_resized)
    X_test_features = extract_features(X_test_resized)

    # Train Random Forest (as an example)
    rf_model, rf_time = train_rf(X_train_features, y_train_labels)
    rf_preds = rf_model.predict(X_test_features)
    rf_acc = accuracy_score(y_test_labels, rf_preds)

    return {"accuracy": rf_acc}

if __name__ == "__main__":
    main()
