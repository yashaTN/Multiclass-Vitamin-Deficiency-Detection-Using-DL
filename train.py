"""
Training Script: ResNet Transfer Learning for Multiclass Vitamin Deficiency Detection.
Executes training loop, validation, metrics calculation, and saves best model checkpoint.
"""

import os
import time
import json
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

from src.config import (
    DATASET_DIR, MODEL_SAVE_PATH, HISTORY_SAVE_PATH, ARTIFACTS_DIR,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, DEVICE, NUM_CLASSES, WEIGHT_DECAY
)
from src.dataset import create_dataloaders
from src.model import build_model
from src.utils import plot_training_history


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/total)*100:.2f}%"})

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

    val_loss = running_loss / max(total, 1)
    val_acc = correct / max(total, 1)
    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser(description="Train ResNet for Vitamin Deficiency Detection")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--max_samples_per_class", type=int, default=150, help="Max images per class (useful for quick balanced CPU/GPU training)")
    parser.add_argument("--architecture", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--save_path", type=str, default=MODEL_SAVE_PATH)
    args = parser.parse_args()

    print(f"============================================================")
    print(f" Multiclass Vitamin Deficiency Detection - Model Training")
    print(f" Architecture : {args.architecture.upper()} (Transfer Learning)")
    print(f" Device       : {DEVICE}")
    print(f" Epochs       : {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f" Max per class: {args.max_samples_per_class}")
    print(f"============================================================")

    # 1. Image Preprocessing & Data Loading
    train_loader, val_loader, train_samples, val_samples = create_dataloaders(
        dataset_dir=DATASET_DIR,
        batch_size=args.batch_size,
        val_split=0.2,
        max_samples_per_class=args.max_samples_per_class
    )
    print(f"[Dataset] Train samples: {len(train_samples)} | Validation samples: {len(val_samples)}")

    # 2. ResNet Transfer Learning Model Setup
    model = build_model(architecture=args.architecture, num_classes=NUM_CLASSES, pretrained=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_epoch": 0,
        "best_val_acc": 0.0
    }

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch}/{args.epochs}] ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Acc: {val_acc*100:.2f}%")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history["best_val_acc"] = float(best_val_acc)
            history["best_epoch"] = epoch
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print(f"  -> Best model checkpoint saved with Val Acc: {best_val_acc*100:.2f}%")

    total_time = time.time() - start_time
    print(f"\n[Finished] Training completed in {total_time/60:.2f} mins.")
    print(f"[Best Result] Epoch {history['best_epoch']} with Validation Accuracy: {history['best_val_acc']*100:.2f}%")

    # If best checkpoint wasn't saved (e.g. 0% val acc fallback), save current state
    if not os.path.exists(args.save_path):
        torch.save(model.state_dict(), args.save_path)

    # Save training history JSON
    with open(HISTORY_SAVE_PATH, "w") as f:
        json.dump(history, f, indent=4)
    print(f"[Saved] Training history saved to: {HISTORY_SAVE_PATH}")

    # Generate and save convergence plot
    plot_path = os.path.join(ARTIFACTS_DIR, "training_convergence.png")
    plot_training_history(history, save_path=plot_path)
    print(f"[Saved] Training loss/accuracy curve plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
