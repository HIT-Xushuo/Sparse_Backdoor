import torch
import torch.nn as nn
import torch.nn.functional as F
from util import *


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        if hasattr(target_layer, "register_full_backward_hook"):
            self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)
        else:
            self.backward_handle = target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, images, class_indices):
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(images)
        score = outputs.gather(1, class_indices.view(-1, 1)).sum()
        score.backward(retain_graph=False)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmaps = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        heatmaps = F.interpolate(
            heatmaps,
            size=(images.shape[2], images.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        flat = heatmaps.flatten(1)
        min_v = flat.min(dim=1)[0].view(-1, 1, 1, 1)
        max_v = flat.max(dim=1)[0].view(-1, 1, 1, 1)
        return (heatmaps - min_v) / (max_v - min_v + 1e-12)


def get_gradcam_layer(model):
    if hasattr(model, "layer4"):
        return model.layer4[-1].conv2
    if hasattr(model, "features"):
        for layer in reversed(model.features):
            if isinstance(layer, nn.Conv2d):
                return layer
    raise ValueError("Please define a Grad-CAM target layer for this architecture.")

def spatial_topk_mask(score, k):
    score = score.reshape(-1)
    idx = torch.topk(score, k=k).indices
    mask = torch.zeros_like(score)
    mask[idx] = 1.0
    return mask.view(32, 32)

def gradcam_masks(model, loader, k, sample_count):
    model.eval()
    gradcam = GradCAM(model, get_gradcam_layer(model))
    masks = []
    heatmaps = []
    seen = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        remain = sample_count - seen
        if remain <= 0:
            break
        images = images[:remain]
        labels = labels[:images.size(0)]

        class_indices = labels

        batch_heatmaps = gradcam.generate(images, class_indices).detach().squeeze(1)
        for heatmap in batch_heatmaps:
            heatmaps.append(heatmap)
            masks.append(spatial_topk_mask(heatmap, k))
        seen += images.size(0)

    gradcam.close()
    masks = torch.stack(masks, dim=0).to(device)
    heatmaps = torch.stack(heatmaps, dim=0).to(device)

    mean_mask = masks.mean(dim=0)
    distances = torch.norm(masks.flatten(1) - mean_mask.flatten().view(1, -1), dim=1)
    representative = masks[torch.argmin(distances)]
    return representative