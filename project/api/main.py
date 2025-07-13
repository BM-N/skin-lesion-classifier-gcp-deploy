import io
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from google.cloud import storage
from PIL import Image, ImageStat
from pydantic import BaseModel

from models.model import get_model
from models.transforms import get_transforms

# configs
METADATA_CSV_FILENAME = "enc_HAM10000_metadata.csv"
TEST_SET_CSV_FILENAME = "test_set.csv"
IMAGE_MANIFEST_FILENAME = "image_manifest.csv"
MODEL_FILENAME = "model.pth"
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
EMBEDDING_CENTER_FILENAME = "embedding_center.pt"
PREDICTIONS_LOG_FILENAME = "predictions_log.csv"
OOD_THRESHOLD = 0.4

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
    true_class: str | None = None


class FeatureExtractor(nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.features = nn.Sequential(*list(full_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


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
        app.state.manifest = pd.read_csv(io.BytesIO(manifest_bytes)).set_index(
            "image_id"
        )

        test_set_bytes = download_blob_to_memory(TEST_SET_CSV_FILENAME)
        app.state.test_set_df = pd.read_csv(io.BytesIO(test_set_bytes))

        model_bytes = download_blob_to_memory(MODEL_FILENAME)
        embedding_center_bytes = download_blob_to_memory(EMBEDDING_CENTER_FILENAME)

        class_names = sorted(df_metadata["dx"].unique())
        app.state.class_names = class_names
        print("Data files and class names loaded successfully.")

        app.state.center_embedding = torch.load(
            io.BytesIO(embedding_center_bytes), map_location=device
        )
        print("Center embedding loaded successfully.")

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
        app.state.feature_extractor = FeatureExtractor(model)

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


def log_prediction_to_gcs(request: Request, log_entry: dict):
    """
    Appends a new prediction log entry to the predictions_log.csv in GCS.
    """
    try:
        bucket = request.app.state.bucket
        blob = bucket.blob(PREDICTIONS_LOG_FILENAME)

        try:
            existing_log_bytes = blob.download_as_bytes()
            df = pd.read_csv(io.BytesIO(existing_log_bytes))
        except Exception:
            df = pd.DataFrame()

        new_entry_df = pd.DataFrame([log_entry])
        df = pd.concat([df, new_entry_df], ignore_index=True)

        output_csv = df.to_csv(index=False)
        blob.upload_from_string(output_csv, "text/csv")
        print("Successfully logged prediction.")

    except Exception as e:
        print(f"ERROR: Failed to log prediction to GCS. {e}")


def transform_image(image_bytes: bytes, true_class: str | None = None) -> torch.Tensor:
    """
    Takes image bytes, applies the project's validation transforms, and returns a tensor.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(device)

        image_stats = ImageStat.Stat(image)
        avg_brightness = np.mean(image_stats.mean)
        image_width, image_height = image.size

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "true_class": true_class,
            "image_width": image_width,
            "image_height": image_height,
            "average_brightness": avg_brightness,
        }
        return tensor, log_entry
    except Exception as e:
        print(f"Error transforming image: {e}")
        raise HTTPException(
            status_code=400, detail="Invalid image file. Could not process."
        )


def get_prediction(request: Request, tensor: torch.Tensor, log_entry: dict):
    """
    Performs inference on the input tensor using the loaded model.
    """
    model = request.app.state.model
    feature_extractor = request.app.state.feature_extractor
    center_embedding = request.app.state.center_embedding
    class_names = request.app.state.class_names 
    tensor = tensor.to(device)

    with torch.no_grad():
        new_embedding = feature_extractor(tensor)
        cos_sim = nn.functional.cosine_similarity(
            new_embedding, center_embedding.unsqueeze(0)
        )
        distance = 1 - cos_sim.item()
        log_entry["ood_distance_score"] = distance

        if distance > OOD_THRESHOLD:
            log_entry["predicted_class"] = "Out-of-Distribution"
            log_entry["confidence_score"] = None
            log_prediction_to_gcs(request, log_entry)
            return {
                "prediction": "Out-of-Distribution",
                "error": "The uploaded image does not appear to be a skin lesion.",
            }
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_index = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_index].item()
    
    predicted_class_abbrev = app.state.class_names[predicted_index]
    predicted_class_full_name = class_names_full[predicted_class_abbrev]
    
    certainty_scores = {
        class_names_full[class_names[i]]: prob.item() for i, prob in enumerate(probabilities)
    }
    
    log_entry["predicted_class"] = predicted_class_full_name
    log_entry["confidence_score"] = confidence
    
    log_prediction_to_gcs(request, log_entry)
    
    return {
        "prediction": predicted_class_full_name,
        "certainty": confidence,
        "all_certainties": certainty_scores,
        "ood_distance_score": distance,
        "error": None
    }

# API endpoints
@app.get("/", summary="Health Check")
def read_root(request: Request):
    if not request.app.state.is_ready:
        raise HTTPException(
            status_code=503, detail="Service not ready. Check startup logs."
        )
    return {"status": "ok", "message": "Welcome to the HAM10000 Classifier API!"}


@app.post("/predict", summary="Classify a Skin Lesion Image")
async def predict(request: Request, file: UploadFile = File(...)):
    """
    The main prediction endpoint. It uses the project's specific modules
    for model architecture and image preprocessing.
    """
    if not request.app.state.is_ready:
        raise HTTPException(
            status_code=503, detail="Service not ready. Check startup logs."
        )
    image_bytes = await file.read()
    tensor, log = transform_image(image_bytes, true_class=None)
    log["true_class"] = None
    result = get_prediction(request, tensor, log)
    return JSONResponse(content=result)


@app.post("/predict-from-url", summary="Classify a skin lesion image from a GCS URL")
async def predict_from_url(request: Request, payload: ImageUrlPayload):
    if not request.app.state.is_ready:
        raise HTTPException(status_code=503, detail="Server resources not available.")
    
    try:
        print(f"Received request to predict from URL: {payload.image_url}")
        if not payload.image_url.startswith(
            f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/"
        ):
            raise HTTPException(status_code=400, detail="Invalid GCS URL format.")

        blob_name = payload.image_url.replace(
            f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/", ""
        )

        blob = request.app.state.bucket.blob(blob_name)
        image_bytes = blob.download_as_bytes()
        tensor, log = transform_image(image_bytes, true_class=payload.true_class)
        result = get_prediction(request, tensor, log)
        return JSONResponse(content=result)
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

        df_merged = df_test.merge(manifest, on="image_id", how="left")
        df_merged.dropna(subset=["relative_path"], inplace=True)

        base_gcs_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}"
        df_merged["image_url"] = df_merged["relative_path"].apply(
            lambda p: f"{base_gcs_url}/{p.replace('\\', '/')}"
        )

        df_merged["dx"] = df_merged["label"].apply(
            lambda label_int: class_names[label_int]
        )
        df_merged["dx_full"] = df_merged["dx"].map(class_names_full)

        result = df_merged[["image_id", "image_url", "dx_full"]].to_dict(
            orient="records"
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@app.get("/monitoring-data", summary="Get prediction log data for the dashboard")
async def get_monitoring_data(request: Request):
    if not request.app.state.is_ready:
        raise HTTPException(status_code=503, detail="Server resources not available.")
    try:
        bucket = request.app.state.bucket
        blob = bucket.blob(PREDICTIONS_LOG_FILENAME)
        if not blob.exists():
            return JSONResponse(content=[])

        log_bytes = blob.download_as_bytes()
        df = pd.read_csv(io.BytesIO(log_bytes))
        return JSONResponse(content=df.to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve monitoring data: {e}"
        )
