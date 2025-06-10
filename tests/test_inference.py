from machine_learning.src.inference import predict_class
import pytest

@pytest.mark.parametrize("model_type", ["cnn", "svm", "rf"])
def test_predict_class_with_models(sample_image_path, model_type):
    result = predict_class(sample_image_path, model_type=model_type)
    assert result in ['glioma', 'meningioma', 'notumor', 'pituitary']
