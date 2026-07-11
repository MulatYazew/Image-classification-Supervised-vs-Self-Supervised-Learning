"""
FoodNet — Grad-CAM
===================
Grad-CAM visual explanations for the custom CNNs.

It hooks the LAST convolutional feature map of "model.backbone" and weights it
by the gradients of the target class, producing a heatmap that shows which part
of the dish drove the prediction. Useful in the report to sanity-check that the
model attends to food regions rather than plates/backgrounds.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM for a Food-251 BaseModel. target_layer defaults to the last
    Conv2d found in model.backbone."""

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model.eval()
        self.target_layer = target_layer or self._last_conv(model)
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    @staticmethod
    def _last_conv(model: nn.Module) -> nn.Module:
        """Pick the last spatial (kernel > 1) conv as the CAM target — the
        backbone's final Conv2d is a 1x1 pointwise mixer with no spatial
        structure, which would yield a flat heatmap. Falls back to the last
        conv of any kind if no k>1 conv exists."""
        last_spatial = None
        last_any = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                last_any = m
                if m.kernel_size[0] > 1:
                    last_spatial = m
        chosen = last_spatial or last_any
        if chosen is None:
            raise ValueError("No Conv2d layer found in the model.")
        return chosen

    def _save_activation(self, module, inp, out) -> None:
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out) -> None:
        self._gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Produce a [0,1] heatmap (HxW) for input_tensor (shape (1,3,H,W)).
        Uses the model's top prediction when class_idx is None."""
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())

        self.model.zero_grad()
        logits[0, class_idx].backward(retain_graph=True)

        grads = self._gradients          # (1, C, h, w)
        acts = self._activations         # (1, C, h, w)
        weights = grads.mean(dim=(2, 3), keepdim=True)     # GAP over spatial dims
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))  # (1,1,h,w)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam

    def overlay(self, image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Blend a heatmap over an RGB image (both H×W×3 / H×W, uint8 output)."""
        import cv2
        heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        if image_rgb.dtype != np.uint8:
            image_rgb = np.uint8(255 * np.clip(image_rgb, 0, 1))
        return np.uint8(alpha * heat + (1 - alpha) * image_rgb)
