import torch
from torchvision.models import resnet50


def get_model(name: str, new_head=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "resnet50":
        model = resnet50(weights="IMAGENET1K_V2")
    if new_head is not None:
        model.fc = new_head
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    return model.to(device)
