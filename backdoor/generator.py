import torch
import torch.nn as nn
import torch.nn.functional as F
from util import *


class AdaptiveScalingFactorGenerator(nn.Module):
    """32x32 generator that predicts a sample-adaptive scalar gamma."""

    def __init__(self):
        super(AdaptiveScalingFactorGenerator, self).__init__()
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.InstanceNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2d(32)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4 * 4 * 32, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

def train_gamma_generator(model, images, target_label, steps, epsilon):
    generator = AdaptiveScalingFactorGenerator().to(device)
    optimizer = torch.optim.AdamW(generator.parameters(), lr=1e-3, weight_decay=1e-4)
    labels = torch.full((images.size(0),), target_label, dtype=torch.long, device=device)
    x_adv = images.clone().detach().to(device)
    gamma_values = []

    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss_x = F.cross_entropy(model(x_adv), labels)
        grad = torch.autograd.grad(loss_x, x_adv, create_graph=False)[0].detach()
        gamma = generator(torch.cat([x_adv.detach(), grad], dim=1))
        gamma_values.append(gamma.detach().mean().item())

        # Targeted update: minimize the cross-entropy to the attacker target.
        x_next = x_adv.detach() - gamma.view(-1, 1, 1, 1) * grad
        x_next = torch.clamp(x_next, images - epsilon, images + epsilon)
        x_next = torch.clamp(x_next, 0, 1)

        optimizer.zero_grad()
        loss_y = F.cross_entropy(model(x_next), labels)
        loss_y.backward()
        optimizer.step()
        x_adv = x_next.detach()

    return generator.eval(), gamma_values