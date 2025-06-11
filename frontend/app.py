import streamlit as st
import requests

# Streamlit page configuration
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor Detection from MRI")
st.write("Upload an MRI image to detect the type of brain tumor:")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # FIXED: use_container_width to avoid deprecation warning
    st.image(uploaded_file, caption="Uploaded MRI Image", use_container_width=True)

    if st.button("Predict"):
        st.write("🔄 Sending image to backend for prediction...")

        try:
            # Prepare image for POST request
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            response = requests.post("https://brain-tumor-detection-9vzr.onrender.com", files=files)

            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: **{result['prediction'].upper()}**")
                if "confidence" in result:
                    st.info(f"Confidence: `{result['confidence']:.2f}%`")
            else:
                st.error(f"Prediction failed: {response.text}")

        except Exception as e:
            st.error(f"Error connecting to backend: {e}")
