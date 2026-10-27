# Deep Learning based Multiclass Vitamin Deficiency Detection for Early Healthcare Intervention

A deep learning clinical decision-support system utilizing **ResNet Transfer Learning** to screen, classify, and explain 11 distinct vitamin and micronutrient deficiencies from photographic manifestations (mucosal, lingual, dermal, ocular, and ungual signs).

---

## 🏗️ System Architecture

The pipeline follows a modular healthcare AI architecture:

```
┌─────────────┐     ┌────────────────────┐     ┌───────────────────────┐
│ Input Image │ ──> │ Image Preprocessing│ ──> │   Transfer Learning   │
│             │     │ (Resize, Normalize)│     │  (ResNet-18 Backbone) │
└─────────────┘     └────────────────────┘     └───────────────────────┘
                                                           │
                                                           ▼
┌──────────────────────────────┐     ┌─────────────────────────────────┐
│  Vitamin Deficiency Class    │ <── │   Classification & Prediction   │
│ Detection & Early Healthcare │     │ (Softmax Probs + Grad-CAM XAI)  │
│         Intervention         │     └─────────────────────────────────┘
└──────────────────────────────┘
```

### 1. Input Image
- Supports clinical image upload, live camera capture, and curated clinical sample datasets.
- Accepts standard formats (`.jpg`, `.jpeg`, `.png`, `.webp`).

### 2. Image Preprocessing
- Format conversion to standard 3-channel RGB (preventing channel mismatch from RGBA/grayscale).
- Spatial normalization: Resized to $224 \times 224$ pixels via bilinear interpolation with center cropping.
- Tensor Z-score Standardization: ImageNet channel distribution ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- Training Data Augmentations: Random crop, random rotation ($\pm 12^\circ$), horizontal flips, and photometric color jitter.

### 3. Transfer Learning (ResNet Backbone)
- Deep Residual Network (`ResNet-18` or `ResNet-50`) initialized with pretrained ImageNet weights.
- Residual skip connections alleviate vanishing gradient dynamics while capturing subtle hierarchical tissue biomarkers.
- Custom classification head:
  - Global Adaptive Average Pooling (`AdaptiveAvgPool2d((1, 1))`)
  - Linear Layer ($512 \to 512$) with Batch Normalization and ReLU
  - Dropout ($p = 0.4$) for regularization
  - Linear Layer ($512 \to 256$) with Batch Normalization and ReLU
  - Dropout ($p = 0.25$)
  - Final Classification Layer ($256 \to 11$)

### 4. Classification & Explainable AI (Grad-CAM)
- Softmax posterior probability distribution across 11 deficiency classes.
- **Grad-CAM (Gradient-weighted Class Activation Mapping)**: Computes gradients of the target class score with respect to the final convolutional feature maps (`layer4`), generating visual heatmaps highlighting the exact anatomical lesions (e.g. tongue surface, lip corners, ocular sclera, hair/nail plate).

### 5. Vitamin Deficiency Class Detection & Early Healthcare Intervention
- Diagnostic finding presentation with confidence scores.
- Clinical manifestation breakdown and targeted anatomical systems.
- Confirmatory diagnostic laboratory test recommendations (e.g. Serum 25(OH)D, Serum B12, MMA, CBC, Ferritin).
- Nutritional therapy recommendations with dietary food sources.
- Red flag alerts for emergency clinical referral.
- Downloadable Clinical Screening Consultation Summary report.

---

## 📁 Dataset & Multiclass Structure

The dataset contains clinical photographs organized across 11 classes:

| Class Name | Nutrient | Characteristic Clinical Manifestation |
| :--- | :--- | :--- |
| **Vitamin A deficiency** | Retinol / Carotenoids | Xerophthalmia, Bitot's spots, keratomalacia, follicular hyperkeratosis |
| **Vitamin B complex deficiency** | General B Vitamins | Fatigue, cheilosis, generalized glossitis, seborrheic dermatitis |
| **Vitamin B-12 deficiency** | Cobalamin | Atrophic Hunter's glossitis, megaloblastic anemia, neuropathy |
| **Vitamin B2 deficiency** | Riboflavin | Angular stomatitis/cheilitis, magenta tongue, corneal vascularization |
| **Vitamin B3 deficieny** | Niacin | Pellagra (Casal's necklace rash, symmetric dermatitis on hands) |
| **Vitamin B9 deficiency** | Folate | Megaloblastic anemia, burning tongue, aphthous ulcers |
| **Vitamin C deficiency** | Ascorbic Acid | Scurvy, swollen bleeding gums, corkscrew hairs, petechiae |
| **Vitamin D deficiency** | Calciferol | Rickets (leg bowing in children), osteomalacia (bone pain in adults) |
| **Vitamin E deficiency** | Tocopherol | Spinocerebellar ataxia, peripheral neuropathy, hemolytic anemia |
| **Vitamin K deficiency** | Phylloquinone | Easy bruising, prolonged coagulation time, mucosal bleeding |
| **zinc, iron, biotin, or protein** | Trace Minerals / Protein | Koilonychia (spoon nails), alopecia, acrodermatitis enteropathica |

---

## 🚀 Quick Start Guide

### 1. Requirements & Environment
The project is built entirely in **Python** using PyTorch, Torchvision, and Streamlit:
```bash
pip install torch torchvision streamlit pillow opencv-python matplotlib seaborn scikit-learn
```

### 2. Launching the Streamlit Web Application
Run the web application directly:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Training the Model
To fine-tune or train the ResNet model on your dataset:
```bash
# Fast training / fine-tuning run
python train.py --epochs 5 --batch_size 16 --max_samples_per_class 100

# Full dataset training
python train.py --epochs 10 --batch_size 32
```
Model checkpoints are saved automatically to `models/resnet_vitamin_detector.pth`.

### 4. Model Evaluation & Confusion Matrix
To evaluate model performance and produce precision/recall/F1 metrics and confusion matrix heatmaps:
```bash
python evaluate.py
```
Evaluation metrics and plots will be saved in `artifacts_eval/`.

---

## 📂 Project Organization

```
vitamin def/
├── dataset/                        # Image dataset partitioned into 11 class folders
├── models/                         # Saved ResNet model weights & training history
│   ├── resnet_vitamin_detector.pth
│   └── training_history.json
├── sample_images/                  # Curated sample test images for 1-click web demo
├── artifacts_eval/                 # Confusion matrix, convergence plots, evaluation metrics
├── src/
│   ├── __init__.py
│   ├── config.py                   # Configuration constants, paths, and class mappings
│   ├── dataset.py                  # PyTorch Dataset, DataLoaders, and preprocessing pipelines
│   ├── model.py                    # ResNet Transfer Learning architecture & Grad-CAM hooks
│   ├── gradcam.py                  # Explainable AI Grad-CAM implementation
│   ├── medical_knowledge.py        # Clinical knowledge base & early intervention protocols
│   └── utils.py                    # Inference helpers, checkpoint loaders, and visualizers
├── train.py                        # Training and validation script
├── evaluate.py                     # Evaluation and classification report generator
├── app.py                          # Streamlit interactive clinical web application
└── README.md                       # Comprehensive documentation
```
