"""
Utility functions for model loading, inference pipeline, and visual analytics.
"""

import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

from .config import (
    MODEL_SAVE_PATH, HISTORY_SAVE_PATH, DEVICE,
    CLASS_NAMES, DISPLAY_NAMES, IMAGE_SIZE
)
from .dataset import get_inference_transforms
from .model import build_model, VitaminResNet


def load_trained_model(model_path: str = MODEL_SAVE_PATH, architecture: str = "resnet18") -> VitaminResNet:
    """
    Loads saved model weights if available, or returns a pretrained transfer-learning
    model ready for immediate inference/fine-tuning.
    """
    model = build_model(architecture=architecture, num_classes=len(CLASS_NAMES), pretrained=True)
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"[INFO] Loaded trained weights from: {model_path}")
        except Exception as e:
            print(f"[WARNING] Failed to load checkpoint ({e}). Using initialized transfer learning weights.")
    else:
        print("[INFO] No trained checkpoint found. Using pretrained ImageNet feature backbone.")
    return model


def predict_single_image(
    model: VitaminResNet,
    image: Image.Image,
    top_k: int = 5
) -> List[Dict]:
    """
    Full inference pipeline adhering to system architecture:
    Input Image -> Image Preprocessing -> ResNet Transfer Learning -> Classification & Probabilities.
    Returns sorted list of Top-K prediction dictionaries.
    """
    model.eval()
    transform = get_inference_transforms(IMAGE_SIZE)
    img_rgb = image.convert("RGB")
    tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_indices = np.argsort(probs)[::-1][:top_k]

    results = []
    for idx in top_indices:
        raw_name = CLASS_NAMES[idx]
        results.append({
            "class_index": int(idx),
            "raw_class_name": raw_name,
            "display_name": DISPLAY_NAMES.get(raw_name, raw_name),
            "probability": float(probs[idx]),
            "percentage": float(probs[idx] * 100.0)
        })

    return results


def get_preprocessing_stages(image: Image.Image) -> Dict[str, Image.Image]:
    """Generates visual representations of intermediate preprocessing stages."""
    rgb = image.convert("RGB")
    resized = rgb.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)

    # Simulated normalized tensor representation
    transform = get_inference_transforms(IMAGE_SIZE)
    tensor = transform(rgb)
    norm_np = tensor.numpy()
    # Normalize channels to 0-255 for display
    norm_vis = np.transpose(norm_np, (1, 2, 0))
    norm_vis = (norm_vis - norm_vis.min()) / (norm_vis.max() - norm_vis.min() + 1e-8)
    norm_img = Image.fromarray(np.uint8(norm_vis * 255))

    return {
        "original": rgb,
        "resized_224": resized,
        "normalized_tensor": norm_img
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str = None):
    """Plots and optionally saves a high-resolution confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[c.split()[0] + (" " + c.split()[1] if len(c.split()) > 1 else "") for c in class_names],
        yticklabels=[c.split()[0] + (" " + c.split()[1] if len(c.split()) > 1 else "") for c in class_names],
        cbar=True
    )
    plt.title("Multiclass Vitamin Deficiency Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("Ground Truth Class", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_training_history(history: Dict, save_path: str = None):
    """Plots Loss and Accuracy training/validation convergence curves."""
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#2563EB", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#DC2626", linewidth=2)
    ax1.set_title("Cross-Entropy Loss Progression", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Accuracy Curve
    ax1.set_title("Cross-Entropy Loss Progression", fontweight="bold", fontsize=12)
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train Acc (%)", color="#10B981", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "s--", label="Val Acc (%)", color="#8B5CF6", linewidth=2)
    ax2.set_title("Classification Accuracy Progression (%)", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
