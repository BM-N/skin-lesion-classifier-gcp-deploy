import io
import os
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from google.cloud import storage
from PIL import Image
from tqdm import tqdm

from models.model import get_model
from models.transforms import get_transforms

GCP_PROJECT_ID = "skin-lesion-classifier"
GCS_BUCKET_NAME = "skin-lesion-classifier-data"
GCS_TRAIN_CSV_PATH = "train_set.csv"
GCS_MODEL_ARTIFACT_PATH = "model.pth"
GCS_IMAGE_MANIFEST_PATH = "image_manifest.csv"
GCS_DESTINATION_BLOB_NAME = "embedding_center.pt"
NUM_SAMPLES = 1000

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


def download_blob_to_memory(bucket, blob_name):
    """Downloads a blob from GCS into an in-memory bytes buffer."""
    try:
        blob = bucket.blob(blob_name)
        return io.BytesIO(blob.download_as_bytes())
    except Exception as e:
        print(f"ERROR: Failed to download {blob_name} from GCS. {e}")
        raise


def download_blob_to_file(bucket, blob_name, destination_file_path):
    """Downloads a blob from GCS to a local file path."""
    try:
        blob = bucket.blob(blob_name)
        blob.download_to_filename(destination_file_path)
    except Exception as e:
        print(f"ERROR: Failed to download {blob_name} from GCS. {e}")
        raise


def upload_to_gcs(bucket, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    try:
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print(
            f"File {source_file_name} uploaded to {destination_blob_name} in bucket {bucket.name}."
        )
    except Exception as e:
        print(f"ERROR: Failed to upload to GCS. {e}")


def create_feature_extractor(model_path, num_classes):
    """
    Loads the full model from a file path and returns the feature extractor.
    """
    new_head = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(p=0.36057091203514374),
        nn.Linear(512, num_classes),
    )
    full_model = get_model(name="resnet50", new_head=new_head)
    full_model.load_state_dict(torch.load(model_path, map_location=device))

    feature_extractor = nn.Sequential(*list(full_model.children())[:-1])
    feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor


def main():
    print("Starting cloud-native embedding center calculation...")

    try:
        storage_client = storage.Client(project=GCP_PROJECT_ID)
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        print(f"Connected to GCS bucket: {GCS_BUCKET_NAME}")

        print("Downloading necessary artifacts from GCS")
        df_train_bytes = download_blob_to_memory(bucket, GCS_TRAIN_CSV_PATH)
        df_manifest_bytes = download_blob_to_memory(bucket, GCS_IMAGE_MANIFEST_PATH)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_model_file:
            download_blob_to_file(bucket, GCS_MODEL_ARTIFACT_PATH, tmp_model_file.name)
            local_model_path = tmp_model_file.name

        df_train = pd.read_csv(df_train_bytes)
        df_manifest = pd.read_csv(df_manifest_bytes)

        if len(df_train) > NUM_SAMPLES:
            df_sample = df_train.sample(n=NUM_SAMPLES, random_state=42)
        else:
            df_sample = df_train
        print(f"Loaded and sampled {len(df_sample)} training records.")

        feature_extractor = create_feature_extractor(local_model_path, num_classes=7)

        preprocess = get_transforms(train=False)
        embeddings = []
        print("Downloading sample images and extracting embeddings...")

        with tempfile.TemporaryDirectory() as temp_image_dir:
            for _, row in tqdm(df_sample.iterrows(), total=df_sample.shape[0]):
                image_id = row["image_id"]

                manifest_row = df_manifest.loc[df_manifest["image_id"] == image_id]
                if manifest_row.empty:
                    continue

                gcs_image_path = manifest_row.iloc[0]["relative_path"]
                local_image_path = os.path.join(temp_image_dir, f"{image_id}.jpg")

                download_blob_to_file(bucket, gcs_image_path, local_image_path)

                image = Image.open(local_image_path).convert("RGB")
                tensor = preprocess(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    embedding = feature_extractor(tensor)
                    embedding = torch.flatten(embedding, 1)
                    embeddings.append(embedding.cpu().numpy())

        if not embeddings:
            print("ERROR: No embeddings were generated.")
            return

        all_embeddings = np.vstack(embeddings)
        center_embedding = torch.tensor(all_embeddings.mean(axis=0))
        print(f"Calculated center embedding with shape: {center_embedding.shape}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_output_file:
            torch.save(center_embedding, tmp_output_file.name)
            local_output_path = tmp_output_file.name

        print(
            f"Successfully saved center embedding to temporary file: {local_output_path}"
        )

        print("\nAttempting to upload final artifact to Google Cloud Storage...")
        upload_to_gcs(bucket, local_output_path, GCS_DESTINATION_BLOB_NAME)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if "local_model_path" in locals() and os.path.exists(local_model_path):
            os.remove(local_model_path)
        if "local_output_path" in locals() and os.path.exists(local_output_path):
            os.remove(local_output_path)
        print("Cleaned up temporary files.")


if __name__ == "__main__":
    main()
