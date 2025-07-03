import io
import os
from contextlib import asynccontextmanager

import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from google.cloud import storage

from data.datamodule import get_loss_class_weights
from models.model import get_model
from models.transforms import get_transforms

# configs
METADATA_CSV_FILENAME = "enc_HAM10000_metadata.csv"
TEST_SET_CSV_FILENAME = "test_set.csv"
IMAGE_MANIFEST_FILENAME = "image_manifest.csv"
MODEL_FILENAME = "model.pth"
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names_full = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions",
}

class ImageUrlPayload(BaseModel):
    image_url: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan. The model is loaded on startup
    and stored in the app's state.
    """
    print("Application startup: Loading model...")
    if not GCS_BUCKET_NAME:
        print("FATAL ERROR: GCS_BUCKET_NAME environment variable not set.")
        app.state.is_ready = False
        yield
        return
    try:
        print(f"Step 1: Downloading resources from GCS bucket: {GCS_BUCKET_NAME}")
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        app.state.bucket = bucket

        def download_blob_to_memory(source_blob_name):
            blob = bucket.blob(source_blob_name)
            return blob.download_as_bytes()

        metadata_bytes = download_blob_to_memory(METADATA_CSV_FILENAME)
        df_metadata = pd.read_csv(io.BytesIO(metadata_bytes))
        
        manifest_bytes = download_blob_to_memory(IMAGE_MANIFEST_FILENAME)
        app.state.manifest = pd.read_csv(io.BytesIO(manifest_bytes)).set_index('image_id')

        test_set_bytes = download_blob_to_memory(TEST_SET_CSV_FILENAME)
        app.state.test_set_df = pd.read_csv(io.BytesIO(test_set_bytes))
        
        class_names = sorted(df_metadata['dx'].unique())
        app.state.class_names = class_names
        print("Data files and class names loaded successfully.")

        model_bytes = download_blob_to_memory(MODEL_FILENAME)        
        new_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.36057091203514374),
            nn.Linear(512, len(app.state.class_names)),
        )
        model = get_model(name="resnet50", new_head=new_head)
        model.load_state_dict(torch.load(io.BytesIO(model_bytes), map_location=device))
        print("Successfuly loaded model from artifact")
        model.to(device)
        model.eval()
        app.state.model = model
        print(f"Model loaded successfully and is running on {device}.")
        app.state.is_ready = True

    except Exception as e:
        print(f"Application startup failed: Could not load resources. {e}")
        app.state.is_ready = False
    
    yield

    print("--- LIFESPAN SHUTDOWN ---")
    app.state.model = None
    app.state.class_names = None
    app.state.manifest = None

app = FastAPI(
    title="Skin Lesion Classifier API (Project-Aware)",
    description="API that uses the fine-tuned model to classify skin lesions.",
    version="2.0.0-gcp-cloud",
    lifespan=lifespan,
)

preprocess = get_transforms(train=False)

def transform_image(image_bytes: bytes) -> torch.Tensor:
    """
    Takes image bytes, applies the project's validation transforms, and returns a tensor.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return preprocess(image).unsqueeze(0)  # type: ignore
    except Exception as e:
        print(f"Error transforming image: {e}")
        raise HTTPException(
            status_code=400, detail="Invalid image file. Could not process."
        )


def get_prediction(model: nn.Module, tensor: torch.Tensor):
    """
    Performs inference on the input tensor using the loaded model.
    """
    tensor = tensor.to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_index = torch.argmax(probabilities).item()
    return probabilities, predicted_index


# API endpoints
@app.get("/", summary="Health Check")
def read_root(request: Request):
    if not request.app.state.is_ready:
        raise HTTPException(status_code=503, detail="Service not ready. Check startup logs.")
    return {"status": "ok", "message": "Welcome to the HAM10000 Classifier API!"}


@app.post("/predict", summary="Classify a Skin Lesion Image")
async def predict(request: Request, file: UploadFile = File(...)):
    """
    The main prediction endpoint. It uses the project's specific modules
    for model architecture and image preprocessing.
    """
    if not request.app.state.is_ready:
        raise HTTPException(status_code=503, detail="Service not ready. Check startup logs.")
    model = request.app.state.model
    class_names = request.app.state.class_names
    if not model or not class_names:
        raise HTTPException(
            status_code=503,
            detail="Model or class names not available. Check server logs.",
        )

    image_bytes = await file.read()
    image_tensor = transform_image(image_bytes)
    probabilities, predicted_index = get_prediction(model, image_tensor)

    predicted_class_abbrev = class_names[predicted_index]
    predicted_class_full_name = class_names_full[predicted_class_abbrev]

    certainty_scores = {
        class_names_full[class_names[i]]: prob.item()
        for i, prob in enumerate(probabilities)
    }

    return JSONResponse(
        content={
            "prediction": predicted_class_full_name,
            "certainty": certainty_scores[predicted_class_full_name],
            "all_certainties": certainty_scores,
        }
    )

@app.post("/predict-from-url", summary="Classify a skin lesion image from a GCS URL")
async def predict_from_url(request: Request, payload: ImageUrlPayload):
    if not request.app.state.is_ready:
        raise HTTPException(status_code=503, detail="Server resources not available.")
    model = request.app.state.model
    class_names = request.app.state.class_names
    if not model or not class_names:
        raise HTTPException(
            status_code=503,
            detail="Model or class names not available. Check server logs.",
        )
    try:
        print(f"Received request to predict from URL: {payload.image_url}")
        if not payload.image_url.startswith(f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/"):
            raise HTTPException(status_code=400, detail="Invalid GCS URL format.")
        
        blob_name = payload.image_url.replace(f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/", "")
        
        blob = request.app.state.bucket.blob(blob_name)
        image_bytes = blob.download_as_bytes()
        image_tensor = transform_image(image_bytes)
        probabilities, predicted_index = get_prediction(model, image_tensor)
        
        predicted_class_abbrev = class_names[predicted_index]
        predicted_class_full_name = class_names_full[predicted_class_abbrev]

        certainty_scores = {
        class_names_full[class_names[i]]: prob.item()
        for i, prob in enumerate(probabilities)
    }
        
        return JSONResponse(content={
            "prediction": predicted_class_full_name,
            "certainty": certainty_scores[predicted_class_full_name],
            "all_certainties": certainty_scores,
        })
    except Exception as e:
        print(f"Error predicting from URL: {e}")
        raise HTTPException(status_code=500, detail="Failed to process image from URL.")

@app.get("/test-images", summary="Get list of test images and labels")
async def get_test_images(request: Request):
    if not request.app.state.is_ready:
        raise HTTPException(status_code=503, detail="Server resources not available.")

    try:
        df_test = request.app.state.test_set_df
        manifest = request.app.state.manifest
        class_names = request.app.state.class_names

        df_merged = df_test.merge(manifest, on='image_id', how='left')
        df_merged.dropna(subset=['relative_path'], inplace=True)

        base_gcs_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}"
        df_merged['image_url'] = df_merged['relative_path'].apply(lambda p: f"{base_gcs_url}/{p.replace('\\', '/')}")
        
        df_merged['dx'] = df_merged['label'].apply(lambda label_int: class_names[label_int])
        df_merged['dx_full'] = df_merged['dx'].map(class_names_full)
        
        result = df_merged[['image_id', 'image_url', 'dx_full']].to_dict(orient="records")
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
