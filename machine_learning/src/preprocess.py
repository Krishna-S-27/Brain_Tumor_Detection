import os
import shutil
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from src.config import RAW_DIR, PROCESSED_DIR, IMG_HEIGHT, IMG_WIDTH, CLASSES, CLASS_TO_INDEX

def load_images_and_labels(data_path):
    data = []
    labels = []
    for label_name in CLASSES:
        folder_path = os.path.join(data_path, label_name)
        label_index = CLASS_TO_INDEX[label_name]
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                data.append(img)
                labels.append(label_index)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    data = np.array(data, dtype='float32') / 255.0
    labels = to_categorical(labels, num_classes=len(CLASSES))
    return data, labels

def save_numpy_data(X_train, X_test, y_train, y_test):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(PROCESSED_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DIR, 'y_test.npy'), y_test)
    print(f"[✔] NumPy arrays saved to {PROCESSED_DIR}")

def prepare_directories():
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)
    for split in ['train', 'test']:
        for cls in CLASSES:
            os.makedirs(os.path.join(PROCESSED_DIR, split, cls), exist_ok=True)

def save_images_to_dir(data_path, split):
    for cls in CLASSES:
        class_path = os.path.join(data_path, cls)
        for img_name in os.listdir(class_path):
            src_path = os.path.join(class_path, img_name)
            dest_path = os.path.join(PROCESSED_DIR, split, cls, img_name)
            try:
                img = cv2.imread(src_path)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                cv2.imwrite(dest_path, img)
            except Exception as e:
                print(f"Failed to process {src_path}: {e}")

def main():
    print("Preparing directory structure...")
    prepare_directories()

    print("Loading and preprocessing training data...")
    X_train, y_train = load_images_and_labels(os.path.join(RAW_DIR, "Training"))

    print("Loading and preprocessing testing data...")
    X_test, y_test = load_images_and_labels(os.path.join(RAW_DIR, "Testing"))

    print("Saving preprocessed images to structured folders...")
    save_images_to_dir(os.path.join(RAW_DIR, "Training"), "train")
    save_images_to_dir(os.path.join(RAW_DIR, "Testing"), "test")

    print("Saving NumPy arrays for model training...")
    save_numpy_data(X_train, X_test, y_train, y_test)

    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    main()
