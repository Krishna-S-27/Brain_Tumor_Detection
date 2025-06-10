import requests

def test_predict_endpoint():
    url = "http://localhost:8000/predict"
    with open("sample_image.jpg", "rb") as img:
        files = {"file": ("sample.jpg", img, "image/jpeg")}
        response = requests.post(url, files=files)

    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
