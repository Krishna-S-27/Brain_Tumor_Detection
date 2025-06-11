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

CNN_MODEL_URL = "https://www.dropbox.com/scl/fi/hycicdjgwyogidwpkmhin/cnn_model.h5?rlkey=6siy5ttso5cxzqb7xwetk8ve5&st=wvpmpeli&dl=1"
SVM_MODEL_URL = "https://www.dropbox.com/scl/fi/89ery357pf19i97tjl3c9/svm_model.pkl?rlkey=m64njjqrarqj7s1lp1wld9m9u&st=9n8m0ry1&dl=1"
RF_MODEL_URL  = "https://www.dropbox.com/scl/fi/rtvw1d9s8v432rip3v4dc/rf_model.pkl?rlkey=bt79zzljcvidel7ns7sxffngr&st=eqfqaye6&dl=1"
