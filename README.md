
# 🧠 Brain Tumor Detection and Classification from MRI Images

This project aims to detect and classify brain tumors using **Machine Learning algorithms** based on MRI scan images. It was developed as part of an academic research and engineering curriculum. The system uses both **deep learning (CNN)** and **classical ML models** to differentiate between four types of brain scans.

---

## 📌 Project Overview

The objective is to automate tumor identification and type classification from MRI images. The categories include:

- Glioma Tumor  
- Meningioma Tumor  
- Pituitary Tumor  
- No Tumor

### 🧠 Why This Matters
Early detection of brain tumors is critical. Automating this process using ML can assist radiologists with faster, accurate, and scalable screening.

---

## 🛠️ Tech Stack

| Category        | Tools / Frameworks                           |
|------------------|----------------------------------------------|
| **Languages**     | Python                                       |
| **ML Libraries**  | Scikit-learn, Keras, TensorFlow, OpenCV      |
| **Visualization** | Matplotlib, Seaborn                         |
| **Testing**       | Pytest / Unittest (System Testing Included) |
| **Deployment**    | Streamlit / Flask (optional)                |

---

## ⚙️ Models Implemented

| Model              | Accuracy   |
|--------------------|------------|
| **CNN (VGG-based)**| 96.0%      |
| **Random Forest**  | 91.0%      |
| **SVM (RBF Kernel)**| ~88.5%     |

Data preprocessing included grayscale conversion, resizing, normalization, and label encoding. Feature extraction was performed using convolutional layers and manual descriptors (for non-CNN models).

---

## 🧪 System Testing

- ✅ Unit tests for data loader and preprocessor  
- ✅ Integration tests for prediction pipeline  
- ✅ Functional test: End-to-end prediction from image upload to output  
- ✅ Error handling tests (invalid image input, missing file, etc.)

Test files are located in the `/tests/` directory and can be run using:

```bash
pytest
```

or

```bash
python -m unittest discover tests/
```

---

## 📥 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/Krishna-S-27/Brain_Tumor_Detection_MachineLearning.git
cd Brain_Tumor_Detection_MachineLearning
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Train or Load Model
If pre-trained model files exist, you can load them. Otherwise, train a model using:
```bash
python train_model.py
```

### 4. Run the App (if Streamlit/Flask frontend is added)
```bash
streamlit run app.py
```

---

## 🧠 Dataset Information

- **Source:** Publicly available MRI datasets (Kaggle, Figshare, or Institutional)  
- **Images:** ~7023 MRI scans  
- **Classes:** Glioma, Meningioma, Pituitary, No Tumor  
- **Split:** 80% Train, 20% Test

---

## 📸 Sample Output

> Add screenshots showing prediction UI or confusion matrix here (if available)

---

## 📫 Contact

For feedback or queries:

- 📧 Email: [krishnashalawadi27@gmail.com](mailto:krishnashalawadi27@gmail.com)  
- 💻 GitHub: [Krishna-S-27](https://github.com/Krishna-S-27)

---

> 🔍 *This project demonstrates how machine learning can assist healthcare through automation and intelligent imaging.*
