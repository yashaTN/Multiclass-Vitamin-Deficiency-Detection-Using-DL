"""
Evaluation Script: Multiclass Vitamin Deficiency Detection.
Computes Confusion Matrix, Precision, Recall, F1-Score, and overall accuracy.
"""

import os
import json
import argparse
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from src.config import (
    DATASET_DIR, MODEL_SAVE_PATH, ARTIFACTS_DIR,
    CLASS_NAMES, DISPLAY_NAMES, DEVICE
)
from src.dataset import create_dataloaders
from src.model import build_model
from src.utils import plot_confusion_matrix


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ResNet Model")
    parser.add_argument("--model_path", type=str, default=MODEL_SAVE_PATH)
    parser.add_argument("--architecture", type=str, default="resnet18")
    parser.add_argument("--max_samples_per_class", type=int, default=100)
    args = parser.parse_args()

    print("============================================================")
    print(" Multiclass Vitamin Deficiency Detection - Evaluation")
    print(f" Model path: {args.model_path}")
    print("============================================================")

    _, val_loader, _, val_samples = create_dataloaders(
        dataset_dir=DATASET_DIR,
        batch_size=32,
        val_split=0.25,
        max_samples_per_class=args.max_samples_per_class
    )

    model = build_model(architecture=args.architecture, num_classes=len(CLASS_NAMES), pretrained=False)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        print("[INFO] Model weights loaded successfully.")
    else:
        print("[WARNING] Model file not found. Loading fresh model for benchmark.")

    y_true, y_pred = evaluate_model(model, val_loader, DEVICE)

    # Classification Report
    target_names = [DISPLAY_NAMES.get(c, c) for c in CLASS_NAMES]
    report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)

    print("\n" + report_text)

    # Save metrics JSON
    metrics_path = os.path.join(ARTIFACTS_DIR, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(report_dict, f, indent=4)
    print(f"[Saved] Metrics saved to: {metrics_path}")

    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, CLASS_NAMES, save_path=cm_path)
    print(f"[Saved] Confusion matrix heatmap saved to: {cm_path}")


if __name__ == "__main__":
    main()
