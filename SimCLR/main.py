import torch.nn as nn
from utlis import load_cifar10_subset
from modules import SimCLRModel
from torchvision.models import resnet18
from torch.utils.data import DataLoader

train_set, test_set = load_cifar10_subset("datasets", 10)

encoder = resnet18()
encoder = nn.Sequential(*list(encoder.children())[:-1])

model = SimCLRModel(encoder, 10)
print(model.eval())

train_loader = DataLoader(train_set, 32, True)