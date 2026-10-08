import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sympy import true
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from util import *
from mask import gradcam_masks
from trigger import optimize_trigger

import models


parser = argparse.ArgumentParser(description='Train Backdoored Model')
parser.add_argument("--surrogate_model", type=str, default="resnet18",
                    choices=["resnet18", "resnet34", "vgg16_bn", "vgg19_bn"])
parser.add_argument("--victim_model", type=str, default="resnet18")
parser.add_argument("--save_surrogate", type=str, default="../save_surrogate")
parser.add_argument("--surrogate_ckpt", type=str, default="benign_model.th")
parser.add_argument("--save_dir", type=str, default="save_mask_ablation")
parser.add_argument("--data_root", type=str, default="../data")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--y_target", type=int, default=0)
parser.add_argument("--poison_rate", type=float, default=0.01)
parser.add_argument("--k", type=int, default=60)
parser.add_argument("--epsilon", type=float, default=8 / 255)
parser.add_argument("--trigger_epochs", type=int, default=200)
parser.add_argument("--gamma_train_steps", type=int, default=100)
parser.add_argument("--mask_samples", type=int, default=128)

args = parser.parse_args()

def main():
    set_random_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    train_dataset = datasets.CIFAR10(
        root=args.data_root,
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )
    test_dataset = datasets.CIFAR10(
        root=args.data_root,
        train=False,
        transform=transforms.ToTensor(),
        download=True,
    )
    clean_test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    variant_dir = args.save_dir
    os.makedirs(variant_dir, exist_ok=True)
    logger = create_logger(variant_dir)
    logger.info("Mask variant: %s")
    logger.info("Config: %s", vars(args))

    surrogate_ckpt = os.path.join(args.save_surrogate, args.surrogate_ckpt)
    surrogate = load_model(args.surrogate_model, 10, surrogate_ckpt)
    surrogate.eval()

    base_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    trigger_images, _ = next(iter(base_loader))
    trigger_images = trigger_images.to(device)

    mask = gradcam_masks(surrogate, base_loader, args.k, args.mask_samples)
    delta, mask_3ch = optimize_trigger(
        surrogate, trigger_images, args.y_target, mask, args.trigger_epochs, args.epsilon, args.gamma_train_steps, logger
    )

    np.save(os.path.join(variant_dir, "delta.npy"), delta.detach().cpu().numpy())
    np.save(os.path.join(variant_dir, "mask.npy"), mask_3ch.detach().cpu().numpy())

    poison_indices = make_poison_indices(train_dataset, args.y_target, args.poison_rate)
    poison_train = build_poison_trainset(train_dataset, poison_indices, args.y_target, delta)
    poison_test = build_poison_testset(test_dataset, args.y_target, delta)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
    ])
    poison_train_loader = DataLoader(
        TensorDatasetWithTransform(poison_train, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    poison_test_loader = DataLoader(
        TensorDatasetWithTransform(poison_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    victim = load_model(args.victim_model, 10)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        victim.parameters(),
        lr=args.lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=5e-4,
    )
    scheduler = MultiStepLR(optimizer, milestones=[60, 90], gamma=0.1)

    logger.info("Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC")

    for epoch in range(args.epochs):
        start = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_step(victim, criterion, optimizer, poison_train_loader)
        clean_loss, clean_acc = test_step(victim, criterion, clean_test_loader)
        poison_loss, poison_acc = test_step(victim, criterion, poison_test_loader)
        scheduler.step()

        logger.info(
            "%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f",
            epoch,
            lr,
            time.time() - start,
            train_loss,
            train_acc,
            poison_loss,
            poison_acc,
            clean_loss,
            clean_acc,
        )

    torch.save(victim.state_dict(), os.path.join(variant_dir, "backdoor_model.th"))



if __name__ == "__main__":
    main()