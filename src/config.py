import os
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_images")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts_eval")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "resnet_vitamin_detector.pth")
HISTORY_SAVE_PATH = os.path.join(MODELS_DIR, "training_history.json")

# Classes as present in the dataset directory
CLASS_NAMES = [
    "Vitamin A deficiency",
    "Vitamin B complex deficiency",
    "Vitamin B-12 deficiency",
    "Vitamin B2 deficiency",
    "Vitamin B3 deficieny",
    "Vitamin B9 deficiency",
    "Vitamin C deficiency",
    "Vitamin D deficiency",
    "Vitamin E deficiency",
    "Vitamin K deficiency",
    "zinc, iron, biotin, or protein deficiency"
]

NUM_CLASSES = len(CLASS_NAMES)

# Human-readable normalized names for display
DISPLAY_NAMES = {
    "Vitamin A deficiency": "Vitamin A Deficiency (Retinol)",
    "Vitamin B complex deficiency": "Vitamin B Complex Deficiency (General)",
    "Vitamin B-12 deficiency": "Vitamin B-12 Deficiency (Cobalamin)",
    "Vitamin B2 deficiency": "Vitamin B2 Deficiency (Riboflavin)",
    "Vitamin B3 deficieny": "Vitamin B3 Deficiency (Niacin / Pellagra)",
    "Vitamin B9 deficiency": "Vitamin B9 Deficiency (Folate / Folic Acid)",
    "Vitamin C deficiency": "Vitamin C Deficiency (Ascorbic Acid / Scurvy)",
    "Vitamin D deficiency": "Vitamin D Deficiency (Calciferol / Rickets)",
    "Vitamin E deficiency": "Vitamin E Deficiency (Tocopherol)",
    "Vitamin K deficiency": "Vitamin K Deficiency (Phylloquinone)",
    "zinc, iron, biotin, or protein deficiency": "Trace Mineral / Protein Deficiency (Zinc, Iron, Biotin)"
}

# Image Preprocessing & Transfer Learning Configuration
IMAGE_SIZE = (224, 224)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10
PATIENCE = 3

# Compute Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
