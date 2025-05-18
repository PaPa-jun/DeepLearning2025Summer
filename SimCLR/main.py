import torch.nn as nn, torch.optim as optim, torch
from utlis import load_cifar10_subset, pretrain, get_augmentations, test_model
from modules import SimCLRModel, Classifier, SimCLRDataset, NTXentCrossEntropyLoss
from torchvision.models import resnet18
from torch.utils.data import DataLoader

train_set, test_set = load_cifar10_subset("datasets", 10, 0.1)

encoder = resnet18()
encoder.fc = nn.Identity()

model = SimCLRModel(encoder, 512).to("cuda:0")

train_set = SimCLRDataset(train_set, get_augmentations())
test_set = SimCLRDataset(test_set, get_augmentations())
train_loader = DataLoader(train_set, 32, True)
test_loader = DataLoader(test_set, 32, True)

optimizer = optim.Adam(model.parameters(), 3e-4)
criterion = NTXentCrossEntropyLoss()

pretrain(model, train_loader, test_loader, optimizer, criterion, 50, 10, "cuda:0")

# pretrained_model = SimCLRModel(encoder, 512)
# pretrained_model.load_state_dict(torch.load("checkpoint.pth", weights_only=False)["model_state_dict"])
# for param in pretrained_model.parameters():
#     param.requires_grad = False

# classifier = Classifier(encoder, out_features=10)
# classifier = classifier.to("cuda:0")

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(classifier.parameters(), 0.001)

# total_loss = 0
# classifier.train()
# for epoch in range(100):
#     total_loss = 0.0
#     total_correct = 0
#     total_samples = 0

#     classifier.train()
#     for images, labels in train_loader:
#         images, labels = images.to("cuda:0"), labels.to("cuda:0")

#         # 前向传播
#         outputs = classifier(images)
#         loss = criterion(outputs, labels)

#         # 反向传播
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # 累计损失和正确率
#         total_loss += loss.item() * images.size(0)
#         predicted = torch.argmax(outputs, dim=1)
#         total_correct += (predicted == labels).sum().item()
#         total_samples += labels.size(0)

#     # 计算平均指标
#     avg_loss = total_loss / total_samples
#     accuracy = total_correct / total_samples

#     # 打印日志
#     print(f"Epoch {epoch+1}/100 | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

# test_loader = DataLoader(test_set, batch_size=32)
# test_model(classifier, test_loader, "cuda:0")
