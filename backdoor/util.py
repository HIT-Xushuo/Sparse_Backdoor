import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

import models

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def set_random_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class TensorDatasetWithTransform(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def load_model(model_name, num_classes, ckpt_path=None):
    model = getattr(models, model_name)(num_classes=num_classes).to(device)
    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location=device)
        if isinstance(state_dict, dict) and "netC" in state_dict:
            state_dict = state_dict["netC"]
        model.load_state_dict(state_dict)
    return model

def make_poison_indices(dataset, target_label, poison_rate):
    shuffle = np.random.permutation(len(dataset))
    total_poison = int(len(dataset) * poison_rate)
    indices = []
    for idx in shuffle:
        if dataset[idx][1] != target_label and len(indices) < total_poison:
            indices.append(idx)
    return set(indices)


def build_poison_trainset(clean_train, poison_indices, target_label, delta):
    poisoned = []
    for idx in range(len(clean_train)):
        image, label = clean_train[idx]
        if idx in poison_indices:
            image = torch.clamp(image.to(device) + delta, 0, 1).detach().cpu()
            label = target_label
        poisoned.append((image, label))
    return poisoned


def build_poison_testset(clean_test, target_label, delta):
    poisoned = []
    for idx in range(len(clean_test)):
        image, label = clean_test[idx]
        if label == target_label:
            continue
        image = torch.clamp(image.to(device) + delta, 0, 1).detach().cpu()
        poisoned.append((image, target_label))
    return poisoned


def train_step(model, criterion, optimizer, loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        pred = outputs.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(loader), float(total_correct) / len(loader.dataset)


def test_step(model, criterion, loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            pred = outputs.data.max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
            total_loss += loss.item()
    return total_loss / len(loader), float(total_correct) / len(loader.dataset)

def create_logger(save_dir):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] - %(message)s", "%Y/%m/%d %H:%M:%S")
    file_handler = logging.FileHandler(os.path.join(save_dir, "output.log"))
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger