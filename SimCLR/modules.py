import torch, torch.nn as nn
from torch.nn.functional import normalize, cross_entropy
from torch.utils.data import Dataset
from torchvision import transforms


class NTXCrossEntropyLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXCrossEntropyLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2: 两个增强视图的投影向量 (batch_size, projection_dim)
        Returns:
            loss: 标量损失值
        """
        # 拼接向量并标准化
        representations = torch.cat([z1, z2], dim=0)  # (2N, D)
        representations = normalize(representations, dim=1)  # 归一化到单位球面

        # 计算相似度矩阵
        similarity_matrix = torch.mm(representations, representations.t())  # (2N, 2N)

        # 生成正样本对的标签
        batch_size = z1.shape[0]
        labels = torch.arange(batch_size, device=z1.device).long()
        labels = torch.cat([labels, labels], dim=0)  # (2N,) 重复两次表示正样本对

        # 计算损失
        logits = similarity_matrix / self.temperature
        logits = (
            logits - torch.eye(logits.shape[0], device=logits.device) * 1e12
        )  # 排除对角线元素
        loss = cross_entropy(logits, labels)  # 交叉熵损失

        return loss


class SimCLRModel(nn.Module):

    def __init__(self, encoder: nn.Module, projection_dim: int):
        """
        SimCLR model.
        """
        super(SimCLRModel, self).__init__()
        self.encoder = encoder
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, projection_dim),
        )

    def forward(self, inputs: torch.Tensor):
        batch_size, channel, height, width = inputs.shape
        features = self.encoder(inputs)
        features = features.view(batch_size, -1)
        projections= self.projection_head(features)
        return features, projections


class SimCLRDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        features, labels = self.dataset[index]
        image_pil = transforms.ToPILImage()(features)
        view_1 = self.transform(image_pil)
        view_2 = self.transform(image_pil)
        return view_1, view_2, labels
