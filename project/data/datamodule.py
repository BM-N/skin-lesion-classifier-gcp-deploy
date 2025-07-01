import os
from collections import Counter

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

class HAM10kDS(Dataset):
    def __init__(
        self, img_dir1, img_dir2, csv_file, transform=None, target_transform=None
    ):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.target_transform = target_transform
        self.folder_paths = [img_dir1, img_dir2]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.df["image_id"].iloc[idx]
        for folder_path in self.folder_paths:
            image_path = os.path.join(folder_path, image_name) + ".jpg"
            if os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
                break
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Could not find image {image_path}")
        label = self.df["label"].iloc[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def get_dataloader(
    img_dir1, img_dir2, csv_file, transform, bs, shuffle=False, *args, **kwargs
):
    ds = HAM10kDS(img_dir1, img_dir2, csv_file, transform)
    return DataLoader(
        ds, batch_size=bs, shuffle=shuffle, pin_memory=True, *args, **kwargs
    )


def get_loss_class_weights(csv_file_path: str, device: str | None = None):
    """
    Calculates inverse frequency class weights for a given CSV file.

    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(csv_file_path)
    class_names = sorted(pd.read_csv('data/enc_HAM10000_metadata.csv')["dx"].unique())
    labels = df["label"].tolist()

    class_counts = Counter(labels)
    # print(f"Class counts from {os.path.basename(csv_file_path)}:{class_counts}")

    loss_weights_raw = {cls: 1.0 / count for cls, count in class_counts.items()}

    class_order = sorted(class_counts.keys())
    sorted_loss_weights = [loss_weights_raw[cls] for cls in class_order]

    class_weights_tensor = torch.tensor(sorted_loss_weights, dtype=torch.float).to(
        device
    )

    return class_weights_tensor, class_order, class_names
