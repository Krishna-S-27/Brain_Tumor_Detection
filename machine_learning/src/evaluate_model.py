import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, roc_curve, auc
)
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from sklearn.preprocessing import label_binarize
import cv2
from src.config import PROCESSED_DIR, MODEL_DIR, CLASSES
from src.utils import plot_confusion_matrix


# Load VGG16 feature extractor
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_model = Model(inputs=vgg_model.input, outputs=vgg_model.output)

def resize_images(X, size=(224, 224)):
    return np.array([cv2.resize(img, size) for img in X])

def extract_features(X):
    features = feature_model.predict(X, batch_size=32, verbose=0)
    return features.reshape(features.shape[0], -1)

def load_test_data():
    X_test = np.load(f"{PROCESSED_DIR}/X_test.npy")
    y_test = np.load(f"{PROCESSED_DIR}/y_test.npy")
    return X_test, y_test

def evaluate_sklearn_model(model_path, X_test, y_test_onehot, model_name):
    y_test = np.argmax(y_test_onehot, axis=1)
    y_test_bin = label_binarize(y_test, classes=range(len(CLASSES)))

    # Resize and preprocess
    X_test_resized = preprocess_input(resize_images(X_test))
    X_test_features = extract_features(X_test_resized)

    model = joblib.load(model_path)
    y_pred = model.predict(X_test_features)
    y_pred_proba = model.predict_proba(X_test_features)

    # Metrics
    print(f"\n[INFO] Classification Report for {model_name.upper()}:")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"[INFO] {model_name.upper()} RMSE: {rmse:.4f}")
    print(f"[INFO] {model_name.upper()} MSE: {mse:.4f}")
    print(f"[INFO] {model_name.upper()} R² Score: {r2:.4f}")

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, labels=CLASSES, path=f"{MODEL_DIR}/confusion_matrix_{model_name}.png")

    # ROC AUC Curve
    plot_roc_curve(y_test_bin, y_pred_proba, model_name)

def evaluate_cnn_model(model_path, X_test, y_test_onehot):
    y_true = np.argmax(y_test_onehot, axis=1)
    y_true_bin = label_binarize(y_true, classes=range(len(CLASSES)))

    model = load_model(model_path)
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n[INFO] Classification Report for CNN:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"[INFO] CNN RMSE: {rmse:.4f}")
    print(f"[INFO] CNN MSE: {mse:.4f}")
    print(f"[INFO] CNN R² Score: {r2:.4f}")

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, labels=CLASSES, path=f"{MODEL_DIR}/confusion_matrix_cnn.png")

    # ROC AUC Curve
    plot_roc_curve(y_true_bin, y_pred_probs, "cnn")

def plot_roc_curve(y_true_bin, y_scores, model_name):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(CLASSES)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i in range(len(CLASSES)):
        plt.plot(fpr[i], tpr[i], label=f"{CLASSES[i]} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name.upper()}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{MODEL_DIR}/roc_curve_{model_name}.png")
    plt.close()

if __name__ == "__main__":
    X_test, y_test = load_test_data()

    print("[INFO] Evaluating SVM Model...")
    evaluate_sklearn_model(f"{MODEL_DIR}/svm_model.pkl", X_test, y_test, model_name="svm")

    print("[INFO] Evaluating Random Forest Model...")
    evaluate_sklearn_model(f"{MODEL_DIR}/rf_model.pkl", X_test, y_test, model_name="rf")

    print("[INFO] Evaluating CNN Model...")
    evaluate_cnn_model(f"{MODEL_DIR}/cnn_model.h5", X_test, y_test)
