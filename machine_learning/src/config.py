import os

# Base and dataset directories
BASE_DIR = r"E:/MINI PROJECT"
RAW_DIR = os.path.join(BASE_DIR, "dataset_mri")          # Original images organized by class folders
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_mri")  # Processed .npy data files
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")       # Directory to save trained models

# Image size (set for VGG16 compatibility)
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Classes (must match folder names in RAW_DIR)
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_TO_INDEX = {cls: idx for idx, cls in enumerate(CLASSES)}
