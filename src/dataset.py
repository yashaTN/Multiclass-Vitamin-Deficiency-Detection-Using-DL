"""
Dataset loader and preprocessing transforms for multiclass vitamin deficiency images.
Adheres to the system architecture step: Image Preprocessing.
"""

import os
from typing import Tuple, List, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from .config import CLASS_NAMES, IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, BATCH_SIZE


def get_train_transforms(image_size: Tuple[int, int] = IMAGE_SIZE):
    """
    Image preprocessing and data augmentation pipeline for training:
    - Resizing and random crop
    - Random horizontal flipping
    - Slight rotation (rotation invariance)
    - Color jitter (photometric variations in lighting/skin tones)
    - Tensor normalization with ImageNet statistics
    """
    return transforms.Compose([
        transforms.Resize((int(image_size[0] * 1.15), int(image_size[1] * 1.15))),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    ])


def get_val_transforms(image_size: Tuple[int, int] = IMAGE_SIZE):
    """
    Deterministic preprocessing pipeline for validation:
    - Resize
    - Center crop
    - Tensor normalization
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    ])


def get_inference_transforms(image_size: Tuple[int, int] = IMAGE_SIZE):
    """Preprocessing pipeline for live inference / web app uploads."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    ])


class VitaminDeficiencyDataset(Dataset):
    """
    Multiclass Vitamin Deficiency PyTorch Dataset.
    Robustly loads images, converts all formats (RGBA/grayscale/CMYK) to RGB,
    and applies preprocessing transforms.
    """
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform=None
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            # Fallback placeholder image if a single file is damaged
            image = Image.new("RGB", IMAGE_SIZE, color=(128, 128, 128))

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        return image_tensor, label, img_path


def scan_dataset(dataset_dir: str, max_samples_per_class: Optional[int] = None) -> List[Tuple[str, int]]:
    """Scans all class subdirectories in dataset_dir and builds (filepath, label_idx) list."""
    samples = []
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_folder = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_folder):
            continue

        class_files = [
            os.path.join(class_folder, f)
            for f in os.listdir(class_folder)
            if f.lower().endswith(valid_exts)
        ]

        if max_samples_per_class is not None and max_samples_per_class > 0 and len(class_files) > max_samples_per_class:
            # Deterministic selection for reproducibility
            class_files = class_files[:max_samples_per_class]

        for filepath in class_files:
            samples.append((filepath, class_idx))

    return samples


def create_dataloaders(
    dataset_dir: str,
    batch_size: int = BATCH_SIZE,
    val_split: float = 0.2,
    max_samples_per_class: Optional[int] = None,
    num_workers: int = 0,
    seed: int = 42
):
    """Creates training and validation DataLoaders with stratified-like split."""
    torch.manual_seed(seed)
    all_samples = scan_dataset(dataset_dir, max_samples_per_class=max_samples_per_class)
    
    if len(all_samples) == 0:
        raise ValueError(f"No valid images found in dataset directory: {dataset_dir}")

    # Random split
    total_count = len(all_samples)
    val_count = int(total_count * val_split)
    train_count = total_count - val_count

    # Shuffle samples deterministically
    import random
    rng = random.Random(seed)
    shuffled = list(all_samples)
    rng.shuffle(shuffled)

    train_samples = shuffled[:train_count]
    val_samples = shuffled[train_count:]

    train_dataset = VitaminDeficiencyDataset(train_samples, transform=get_train_transforms())
    val_dataset = VitaminDeficiencyDataset(val_samples, transform=get_val_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, train_samples, val_samples
