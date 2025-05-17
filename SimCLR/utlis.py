import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Subset, DataLoader
from modules import SimCLRDataset


def load_cifar10_subset(path, subset_classes=10, train_percent=0.1, seed=42):
    """
    加载CIFAR-10子集数据（不含验证集）
    Args:
        path: 数据集路径
        subset_classes: 使用的类别数量（前 n 类）
        train_percent: 从训练集中采样的比例
        seed: 随机种子
    Returns:
        train_dataset: 训练集子集
        test_dataset: 测试集（完整测试集）
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.CIFAR10(
        root=path, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=path, train=False, download=True, transform=transform
    )

    train_indices = [
        i for i, (_, label) in enumerate(train_dataset) if label < subset_classes
    ]
    test_indices = [
        i for i, (_, label) in enumerate(test_dataset) if label < subset_classes
    ]

    num_samples = int(train_percent * len(train_indices))
    sampled_indices = np.random.choice(train_indices, num_samples, replace=False)

    train_subset = Subset(train_dataset, sampled_indices)
    test_subset = Subset(test_dataset, test_indices)

    return SimCLRDataset(train_subset, get_augmentations()), SimCLRDataset(
        test_subset, get_augmentations()
    )


def get_augmentations():
    return transforms.Compose(
        [
            # 1. 随机裁剪并缩放（空间增强）
            transforms.RandomResizedCrop(
                size=32,
                scale=(0.2, 1.0),
                ratio=(0.75, 1.33),
                interpolation=InterpolationMode.BILINEAR,
            ),
            # 2. 颜色抖动（颜色增强）
            transforms.ColorJitter(
                brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2
            ),
            # 3. 随机水平翻转
            transforms.RandomHorizontalFlip(p=0.5),
            # 4. 转换为张量
            transforms.ToTensor(),
            # 5. 归一化
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
            ),
        ]
    )


def pretrain_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str = "cpu",
):
    model.train()
    total_loss = 0
    for view_1, view_2, _ in train_loader:
        view_1 = view_1.to(device)
        view_2 = view_2.to(device)

        _, projections_1 = model(view_1)
        _, projections_2 = model(view_2)

        loss = criterion(projections_1, projections_2)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(train_loader)
    return average_loss


def pretrain(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epochs: int = 10,
    device: str = "cpu",
):
    best_loss = torch.inf
    for epoch in range(epochs):
        loss = pretrain_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch: {epoch + 1}/{epochs} | Loss: {loss:.4f}")
        if loss < best_loss:
            torch.save(model, "pretrained_model.pth")
            print(f'Best model saved as "pretrained_model.pth" with loss: {loss:.4f}.')
