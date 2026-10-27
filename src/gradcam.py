"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Explainable AI (XAI)
in Multiclass Vitamin Deficiency Detection.
Produces interpretable heatmaps highlighting clinical biomarkers.
"""

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2

class GradCAM:
    """Computes Grad-CAM visualizations for ResNet models."""
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None):
        """
        Generates Grad-CAM heatmap for a single input tensor of shape (1, 3, H, W).
        If target_class is None, uses the highest scoring predicted class.
        """
        input_tensor = input_tensor.clone()
        input_tensor.requires_grad = True

        # Forward pass
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = torch.argmax(probs, dim=1).item()

        score = logits[0, target_class]

        # Backward pass for target class score
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Retrieve gradients and activations from model hook
        gradients = self.model.get_activations_gradient()
        activations = self.model.get_activations()

        if gradients is None or activations is None:
            # Fallback if hooks were not triggered
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3])), target_class, probs[0].detach().cpu().numpy()

        # Global average pooling of gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # Weight the feature channels
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)

        # Normalize heatmap to [0, 1]
        heatmap_np = heatmap.detach().cpu().numpy()
        max_val = np.max(heatmap_np)
        if max_val > 0:
            heatmap_np = heatmap_np / max_val
        else:
            heatmap_np = np.zeros_like(heatmap_np)

        return heatmap_np, target_class, probs[0].detach().cpu().numpy()

    def overlay_heatmap(self, original_pil_image: Image.Image, heatmap_np: np.ndarray, alpha: float = 0.5) -> Image.Image:
        """
        Resizes heatmap to original image dimensions, applies Jet colormap,
        and overlays on top of original RGB image.
        """
        orig_w, orig_h = original_pil_image.size
        # Resize heatmap
        resized_cam = cv2.resize(heatmap_np, (orig_w, orig_h))
        heatmap_uint8 = np.uint8(255 * resized_cam)
        
        # Color map
        colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

        orig_np = np.array(original_pil_image.convert("RGB"))
        blended = np.uint8(alpha * colormap_rgb + (1 - alpha) * orig_np)

        return Image.fromarray(blended)
