from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from machine_learning.src.inference import predict_class

app = FastAPI()

# Allow all CORS (for Streamlit access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create folder for uploads
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Query("cnn", enum=["cnn", "svm", "rf"])
):
    """
    Predict brain tumor class from an MRI image using selected model (cnn, svm, rf).
    """
    try:
        # Save uploaded file
        file_location = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Call the prediction function
        prediction = predict_class(file_location, model_name)

        return {"filename": file.filename, "model": model_name, "prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
