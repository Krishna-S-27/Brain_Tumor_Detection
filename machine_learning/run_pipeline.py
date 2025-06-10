# run_pipeline.py

from src.preprocess import load_data
from src.train_model import train_svm, train_rf, train_cnn
from src.evaluate_model import evaluate_sklearn_model, evaluate_cnn_model
import numpy as np

print("[INFO] Loading data...")
X_train, X_test, y_train, y_test = load_data()

print("[INFO] Flattening data for ML models...")
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
y_train_labels = np.argmax(y_train, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

print("[INFO] Training SVM...")
svm_model = train_svm(X_train_flat, y_train_labels)
print("[INFO] Evaluating SVM...")
evaluate_sklearn_model("models/svm_model.pkl", X_test_flat, y_test_labels)

print("[INFO] Training Random Forest...")
rf_model = train_rf(X_train_flat, y_train_labels)
print("[INFO] Evaluating Random Forest...")
evaluate_sklearn_model("models/rf_model.pkl", X_test_flat, y_test_labels)

print("[INFO] Training CNN...")
cnn_model = train_cnn(X_train, y_train, X_test, y_test)
print("[INFO] Evaluating CNN...")
evaluate_cnn_model("models/cnn_model.h5", X_test, y_test)
