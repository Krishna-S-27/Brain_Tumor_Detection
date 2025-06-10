import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input

# Create feature extractor model using VGG16
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_model = Model(inputs=vgg_model.input, outputs=vgg_model.output)

def extract_features(X):
    # Preprocess images as required by VGG16
    X = preprocess_input(X)
    features = feature_model.predict(X, batch_size=32, verbose=1)
    return features.reshape(features.shape[0], -1)
