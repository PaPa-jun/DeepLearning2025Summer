import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from torch.optim.lr_scheduler import CosineAnnealingLR


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

    return train_subset, test_subset


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
            # 2. 随机水平翻转（概率 50%）
            transforms.RandomHorizontalFlip(p=0.5),
            # 3. 颜色抖动（整体概率 80%，参数强度与论文一致）
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.4,
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ]
    )


# 修改点：保存完整训练状态
def save_checkpoint(model, optimizer, epoch, loss, path="checkpoint.pth"):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
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

    return total_loss / len(train_loader)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu",
):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for view_1, view_2, _ in val_loader:
            view_1 = view_1.to(device)
            view_2 = view_2.to(device)

            _, projections_1 = model(view_1)
            _, projections_2 = model(view_2)

            loss = criterion(projections_1, projections_2)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def pretrain(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epochs: int = 10,
    patience=10,
    device: str = "cpu",
):
    best_loss = torch.inf
    no_improve = 0
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        train_loss = pretrain_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch: {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss)
            print("best model saved.")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break


def test_model(model, test_loader, device="cuda"):
    model.eval()  # 设置为评估模式 [[1]]
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 禁用梯度计算 [[1]]
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)

            # 统计指标
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = total_correct / total_samples
    print(f"Test Accuracy: {accuracy:.4f}")

    # 混淆矩阵可视化 [[7]]
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("test_confusion_matrix.png")
    plt.close()

    # 分类报告 [[7]]
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
