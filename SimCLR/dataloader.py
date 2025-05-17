import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from torchvision.transforms.functional import InterpolationMode


def load_cifar10_subset(path, subset_classes=10, train_percent=0.1, seed=42):
    """
    加载CIFAR-10子集数据（不含验证集）
    参数:
        path: 数据集路径
        subset_classes: 使用的类别数量（前 n 类）
        train_percent: 从训练集中采样的比例
        seed: 随机种子
    返回:
        train_dataset: 训练集子集
        test_dataset: 测试集（完整测试集）
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    transform = transforms.Compose([transforms.ToTensor()])

    # 加载 CIFAR-10 数据集
    train_dataset = datasets.CIFAR10(
        root=path, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=path, train=False, download=True, transform=transform
    )

    # 筛选前 subset_classes 类的样本
    train_indices = [
        i for i, (_, label) in enumerate(train_dataset) if label < subset_classes
    ]
    test_indices = [
        i for i, (_, label) in enumerate(test_dataset) if label < subset_classes
    ]

    # 从训练集中采样
    num_samples = int(train_percent * len(train_indices))
    sampled_indices = np.random.choice(train_indices, num_samples, replace=False)
    train_subset = Subset(train_dataset, sampled_indices)

    # 创建测试集子集
    test_subset = Subset(test_dataset, test_indices)

    return train_subset, test_subset


train_data, test_data = load_cifar10_subset(
    path="datasets", subset_classes=10, train_percent=0.1
)

print(f"训练集大小: {len(train_data)}")
print(f"测试集大小: {len(test_data)}")


# # 数据增强定义
# def get_augmentations(normalize=True):
#     """
#     定义SimCLR数据增强
#     参数:
#         normalize: 是否添加Normalize
#     返回:
#         augmentation: 数据增强操作
#     """
#     transform_list = [
#         # 1. 随机调整大小并裁剪到32x32
#         transforms.RandomResizedCrop(size=32),
#         # 2. 以0.5的概率水平翻转图像
#         transforms.RandomHorizontalFlip(p=0.5),
#         # 3. 以0.8的概率应用颜色抖动
#         # 4. 以0.2的概率转换为灰度图
#         transforms.ToTensor(),
#     ]
#     if normalize:
#         transform_list.append(
#             transforms.Normalize(  # cifar-10数据集的均值和标准差
#                 (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
#             )
#         )
#     return transforms.Compose(transform_list)


def get_augmentations(normalize: bool = True):
    transform_list = [
        transforms.RandomResizedCrop(
            size=32,
            scale=(0.2, 1.0),
            ratio=(0.75, 1.33),
            interpolation=InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.2,
                )
            ],
            p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
    ]
    if normalize is True:
        transform_list.append(
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        )
    return transforms.Compose(transform_list)


# SimCLR数据集包装器
class SimCLRDatasetWrapper(torch.utils.data.Dataset):
    """
    数据集包装器，为SimCLR生成两个增强视图
    """

    def __init__(self, dataset, normalize=True):
        self.dataset = dataset
        self.augment = get_augmentations(normalize=normalize)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image_pil = transforms.ToPILImage()(image)
        view1 = self.augment(image_pil)
        view2 = self.augment(image_pil)
        return (view1, view2), label

    def __len__(self):
        return len(self.dataset)


# 可视化增强视图
def show_augmentations(dataset, index):
    """
    显示数据集的增强视图
    """
    (view1, view2), label = dataset[index]
    raw_image = dataset.dataset[index][0]

    # 将 view1、view2 和 raw_image 都转换为 numpy 格式并 transpose
    def to_numpy_image(tensor):
        image = tensor.numpy()
        image = np.transpose(image, (1, 2, 0))  # C,H,W -> H,W,C
        return np.clip(image, 0, 1)  # 防止显示报错

    view1_np = to_numpy_image(view1)
    view2_np = to_numpy_image(view2)
    raw_np = to_numpy_image(raw_image)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(raw_np)
    plt.title("raw_image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(view1_np)
    plt.title("view 1")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(view2_np)
    plt.title("view 2")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# 包装为 SimCLR 数据集，不进行 normalize（为了可视化）
simclr_train = SimCLRDatasetWrapper(train_data, normalize=False)

# 显示增强视图
show_augmentations(simclr_train, 0)
