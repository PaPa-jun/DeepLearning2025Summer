import torch
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm

############################################################
# 模型定义
############################################################

# 参考 LeNet-5 网络
class CNN(nn.Module):
    def __init__(self, kernel_size=5, num_conv_layers=3, batch_size=16):
        super(CNN, self).__init__()
        
        # 参数校验
        assert num_conv_layers in [1, 2, 3], "num_conv_layers must be 1, 2, or 3"
        if kernel_size % 2 == 0:
            kernel_size += 1  # 保证为奇数
        
        # 通道数配置
        channels = [6, 16, 32][:num_conv_layers]
        
        # 构建卷积层
        conv_layers = []
        in_channels = 1
        
        for i in range(num_conv_layers):
            # 调整padding策略，确保输出尺寸正确
            padding = kernel_size // 2 if i < num_conv_layers - 1 else 0
            conv_layers.extend([
                nn.Conv2d(in_channels, channels[i], kernel_size, padding=padding),
                nn.Sigmoid(),
                nn.AvgPool2d(2, 2)
            ])
            in_channels = channels[i]
        
        self.conv_net = nn.Sequential(*conv_layers)
        
        # 动态计算全连接层输入尺寸
        with torch.no_grad():
            dummy = torch.randn(batch_size, 1, 28, 28)
            dummy_out = self.conv_net(dummy)
            fc_input_size = dummy_out.view(dummy_out.size(0), -1).size(1)
        
        # 全连接层
        self.fc_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
        )
    
    def forward(self, x):
        x = self.conv_net(x)
        y = self.fc_net(x)
        return y


# 加入了 BatchNorm 层和 Dropout 层以改善过拟合
class BetterCNN(nn.Module):
    def __init__(self, kernel_size=5, num_conv_layers=3, batch_size=32, dropout_rate=0.3):
        super(BetterCNN, self).__init__()
        
        # 参数校验
        assert num_conv_layers in [1, 2, 3], "num_conv_layers must be 1, 2, or 3"
        if kernel_size % 2 == 0:
            kernel_size += 1  # 保证为奇数
        
        # 通道数配置
        channels = [6, 16, 32][:num_conv_layers]
        
        # 构建卷积层
        conv_layers = []
        in_channels = 1
        
        for i in range(num_conv_layers):
            padding = kernel_size // 2 if i < num_conv_layers - 1 else 0
            conv_layers.extend([
                nn.Conv2d(in_channels, channels[i], kernel_size, padding=padding),
                nn.BatchNorm2d(channels[i]),  # 添加批归一化
                nn.Sigmoid(),
                nn.Dropout2d(dropout_rate/2),  # 卷积层使用较低的dropout率
                nn.AvgPool2d(2, 2)
            ])
            in_channels = channels[i]
        
        self.conv_net = nn.Sequential(*conv_layers)
        
        # 动态计算全连接层输入尺寸
        with torch.no_grad():
            dummy = torch.randn(batch_size, 1, 28, 28)
            dummy_out = self.conv_net(dummy)
            fc_input_size = dummy_out.view(dummy_out.size(0), -1).size(1)
        
        # 全连接层
        self.fc_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, 120),
            nn.BatchNorm1d(120),  # 添加批归一化
            nn.Sigmoid(),
            nn.Dropout(dropout_rate),  # 全连接层使用较高的dropout率
            
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),  # 添加批归一化
            nn.Sigmoid(),
            nn.Dropout(dropout_rate),  # 全连接层使用较高的dropout率
            
            nn.Linear(84, 10),
        )
    
    def forward(self, x):
        x = self.conv_net(x)
        y = self.fc_net(x)
        return y

############################################################
# 加载数据集
############################################################

# 设置随机种子
torch.manual_seed(42)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载完整数据集
train_dataset = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', 
    train=False, 
    transform=transform
)

# 随机抽取1%数据
def get_subset(dataset, ratio=0.01):
    indices = np.random.choice(
        len(dataset), 
        size=int(len(dataset)*ratio), 
        replace=False
    )
    return Subset(dataset, indices)

train_subset = get_subset(train_dataset)
test_subset = get_subset(test_dataset)

############################################################
# 定义一些工具函数
############################################################

# 模型训练，采用交叉熵损失和 Adam 优化器，K-折交叉验证
def train_model(
    model,
    train_subset,
    num_epochs=10,
    batch_size=64,
    n_splits=5,
    lr=0.001,
    device='cuda',
    verbose=True,
    patience=3,
    min_delta=0.001
):
    """
    使用 KFold 交叉验证训练模型，返回全局最佳模型实例
    
    参数:
        model: 已初始化的模型实例
        train_subset: 训练数据子集（Subset）
        num_epochs: 训练轮数
        batch_size: 批次大小
        n_splits: KFold 折数
        lr: 学习率
        device: 训练设备
        verbose: 是否打印训练信息
        patience: 早停耐心值
        min_delta: 验证损失的最小改善阈值
    
    返回:
        all_train_losses: 所有折的训练损失 (list of lists)
        all_val_losses: 所有折的验证损失 (list of lists)
        best_model: 全局最佳模型实例（深拷贝）
        best_val_loss: 全局最佳验证损失
    """
    # 初始化 KFold
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 存储所有折的损失
    all_train_losses = []
    all_val_losses = []
    
    # 全局最佳模型跟踪
    best_val_loss = float('inf')
    best_model = None
    
    # 交叉验证循环 - 添加外部进度条
    fold_iterator = tqdm(enumerate(kfold.split(train_subset)), 
                        total=n_splits,
                        desc="KFold Progress",
                        disable=not verbose)
    
    for fold, (train_ids, val_ids) in fold_iterator:
        if verbose:
            print(f'\nFOLD {fold + 1}/{n_splits}')
            print('-----------------------')
        
        # 克隆模型
        fold_model = copy.deepcopy(model).to(device)
        optimizer = optim.Adam(fold_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # 创建数据子集
        fold_train = Subset(train_subset, train_ids)
        fold_val = Subset(train_subset, val_ids)
        
        # 数据加载器
        train_loader = DataLoader(fold_train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(fold_val, batch_size=batch_size, shuffle=False)
        
        # 当前折的损失记录
        fold_train_losses = []
        fold_val_losses = []
        
        # 早停相关变量
        no_improvement = 0
        best_fold_val_loss = float('inf')
        
        # 训练循环 - 添加epoch进度条
        epoch_iterator = tqdm(range(num_epochs),
                             desc=f"Fold {fold+1} Epochs",
                             leave=False,
                             disable=not verbose)
        
        for epoch in epoch_iterator:
            fold_model.train()
            running_train_loss = 0.0
            
            # 训练阶段 - 添加batch进度条
            batch_iterator = tqdm(train_loader,
                                 desc=f"Fold {fold+1} Training",
                                 leave=False,
                                 disable=not verbose)
            
            for images, labels in batch_iterator:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = fold_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * images.size(0)
                
                # 更新batch进度条信息
                batch_iterator.set_postfix({"Batch Loss": loss.item()})
            
            # 计算平均训练损失
            epoch_train_loss = running_train_loss / len(fold_train)
            fold_train_losses.append(epoch_train_loss)
            
            # 验证阶段
            fold_model.eval()
            running_val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = fold_model(images)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item() * images.size(0)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            # 计算验证指标
            epoch_val_loss = running_val_loss / len(fold_val)
            fold_val_losses.append(epoch_val_loss)
            val_accuracy = 100 * correct / total
            
            # 更新全局最佳模型
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model = copy.deepcopy(fold_model)
            
            # 早停逻辑
            if epoch_val_loss < best_fold_val_loss - min_delta:
                best_fold_val_loss = epoch_val_loss
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    if verbose:
                        print(f'Early stopping triggered after {epoch+1} epochs!')
                    break
            
            # 更新epoch进度条信息
            epoch_iterator.set_postfix({
                "Train Loss": f"{epoch_train_loss:.4f}",
                "Val Loss": f"{epoch_val_loss:.4f}",
                "Val Acc": f"{val_accuracy:.2f}%",
                "No Imp": f"{no_improvement}/{patience}"
            })
        
        # 保存当前折的损失
        all_train_losses.append(fold_train_losses)
        all_val_losses.append(fold_val_losses)
        
        # 更新fold进度条信息
        fold_iterator.set_postfix({
            "Best Val Loss": f"{best_fold_val_loss:.4f}",
            "Global Best": f"{best_val_loss:.4f}"
        })
    
    if verbose:
        print(f'\nGlobal Best Val Loss: {best_val_loss:.4f}')
    
    return all_train_losses, all_val_losses, best_model, best_val_loss

# 在测试集上评估模型性能
def evaluate_model(
    model,  # 模型实例
    test_subset,
    batch_size=64,
    device='cuda',
    verbose=True,
):
    """
    评估模型在测试集上的表现
    
    参数:
        model: 模型实例（可直接传入训练后的最佳模型）
        test_subset: 测试数据子集（Subset）
        batch_size: 批次大小
        device: 设备
        verbose: 是否打印结果
    
    返回:
        test_loss: 测试集损失
        test_accuracy: 测试集准确率（%）
    """
    model.to(device)
    model.eval()
    
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = test_loss / len(test_subset)
    test_accuracy = 100 * correct / total
    
    if verbose:
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.2f}%")
    
    return test_loss, test_accuracy

# 绘制损失函数图像
def plot_all_folds(train_losses, val_losses):
    """
    绘制所有折的训练和验证损失曲线（支持早停导致的变长序列）
    
    参数:
        train_losses: 所有折的训练损失（list of lists，可能长度不同）
        val_losses: 所有折的验证损失（list of lists，可能长度不同）
    """
    plt.figure(figsize=(12, 6))
    n_folds = len(train_losses)
    max_epochs = max(len(losses) for losses in train_losses)  # 获取最大epoch数
    
    # 绘制每一折的曲线
    for i in range(n_folds):
        epochs = range(1, len(train_losses[i]) + 1)
        plt.plot(epochs, train_losses[i], label=f'Train Fold {i+1}', linestyle='--', alpha=0.5)
        plt.plot(epochs, val_losses[i], label=f'Val Fold {i+1}', linestyle='-', alpha=0.5)
    
    # 计算并绘制全局平均损失（对齐到最短长度）
    min_length = min(len(losses) for losses in train_losses)
    aligned_train = [losses[:min_length] for losses in train_losses]
    aligned_val = [losses[:min_length] for losses in val_losses]
    
    avg_train_loss = np.mean(aligned_train, axis=0)
    avg_val_loss = np.mean(aligned_val, axis=0)
    
    plt.plot(range(1, min_length + 1), avg_train_loss, 
             label='Avg Train Loss (aligned)', linestyle='--', linewidth=2, color='black')
    plt.plot(range(1, min_length + 1), avg_val_loss, 
             label='Avg Val Loss (aligned)', linestyle='-', linewidth=2, color='darkred')
    
    plt.title('Training and Validation Loss Across All Folds (with Early Stopping)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim(1, max_epochs)  # 设置x轴范围
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 将图例放在图表外右侧
    plt.grid(True)
    plt.tight_layout()  # 自动调整布局
    plt.show()

############################################################
# 实验主要流程代码
############################################################

# model = CNN(kernel_size=3, num_conv_layers=1, batch_size=32)
model = BetterCNN(kernel_size=7, num_conv_layers=1, batch_size=32, dropout_rate=0.5)
all_train_loss, all_val_loss, best_model, best_val_loss = train_model(model, train_subset, 500, 16, 5, 0.001, 'mps', True, patience=10)

test_loss, test_acc = evaluate_model(
    model=best_model,  # 直接传入模型实例
    test_subset=test_subset,
    device="mps"
)

plot_all_folds(all_train_loss, all_val_loss)