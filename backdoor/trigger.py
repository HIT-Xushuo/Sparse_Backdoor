import torch
import torch.nn as nn
import torch.nn.functional as F
from generator import train_gamma_generator
from util import *

def optimize_trigger(model,
                     images,
                     target_label,
                     mask,
                     trigger_epochs,
                     epsilon,
                     gamma_train_steps,
                     logger):
    model.eval()
    images = images.to(device)
    labels = torch.full((images.size(0),), target_label, dtype=torch.long, device=device)
    mask = mask.unsqueeze(0).repeat(3, 1, 1).to(device)
    delta = torch.zeros(3, 32, 32, device=device)

    gamma_generator, gamma_history = train_gamma_generator(
        model, images, target_label, gamma_train_steps, epsilon
    )


    for epoch in range(trigger_epochs):
        batch_delta = delta.detach().clone().unsqueeze(0).requires_grad_(True)
        poisoned = torch.clamp(images + batch_delta, 0, 1)
        loss = F.cross_entropy(model(poisoned), labels)
        grad_delta, grad_image = torch.autograd.grad(loss, [batch_delta, poisoned])

        with torch.no_grad():
            gamma = gamma_generator(torch.cat([poisoned.detach(), grad_image.detach()], dim=1))
            step = gamma.mean().item()
            delta = delta - step * grad_delta.squeeze(0)
            delta = torch.clamp(delta, -epsilon, epsilon)
            delta = delta * mask

        if epoch % 20 == 0 or epoch == trigger_epochs - 1:
            logger.info(
                "Trigger epoch %d/%d",
                epoch,
                trigger_epochs,
            )

    return delta.detach(), mask.detach()