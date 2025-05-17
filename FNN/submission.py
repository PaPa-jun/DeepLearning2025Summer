import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_openml

##############################################################################
# 定义模型
##############################################################################
class FNN(nn.Module):
    def __init__(self, input_size=13, hidden_size=64, output_size=1, num_hidden_layers=1, activate_function="ReLU"):
        super(FNN, self).__init__()
        
        # 动态构建隐藏层
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))  # 输入层到第一个隐藏层
        layers.append(nn.ReLU())  # 激活函数
        
        # 添加额外的隐藏层
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if activate_function == "ReLU":
                layers.append(nn.ReLU())
            elif activate_function == "Tanh":
                layers.append(nn.Tanh())
            elif activate_function == "Sigmoid":
                layers.append(nn.Sigmoid())
            else: raise(ValueError("无效的激活函数，必须是 ['ReLU', 'Tanh', 'Sigmoid']."))
        
        # 输出层
        layers.append(nn.Linear(hidden_size, output_size))
        
        # 将所有层组合成一个 Sequential 模块
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
##############################################################################
# 定义训练和测试函数
##############################################################################
    
def train_model(model, X, y, num_epochs=100, lr=0.001, batch_size=32, device='cpu', val_ratio=0.2, patience=50):
    """
    训练FNN模型并返回训练/验证损失列表
    参数:
        model: 待训练模型
        X: 输入特征列表/数组
        y: 目标值列表/数组
        num_epochs: 训练轮数
        lr: 学习率
        batch_size: 批量大小
        device: 训练设备 ('cpu' 或 'cuda')
        val_ratio: 验证集比例
        patience: 早停耐心值，即在验证损失不再改善时等待的epoch数
    返回:
        train_losses: 各epoch训练损失列表
        val_losses: 各epoch验证损失列表
    """
    # 转换数据为PyTorch张量
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_tensor, y_tensor, test_size=val_ratio, random_state=42
    )

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    # 初始化模型和优化器
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * inputs.size(0)
        
        # 计算平均训练损失
        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        with torch.no_grad():
            inputs, targets = next(iter(val_loader))
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            val_loss = criterion(outputs, targets)
            avg_val_loss = val_loss.item()
            val_losses.append(avg_val_loss)

            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), 'best_model.pth')
                epochs_no_improve = 0  # 重置计数器
            else:
                epochs_no_improve += 1  # 验证损失没有改善

        # 打印训练进度
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] | '
                  f'Train Loss: {avg_train_loss:.4f} | '
                  f'Val Loss: {avg_val_loss:.4f} | '
                  f'Best Val Loss: {best_val_loss:.4f} | '
                  f'Patience: {patience - epochs_no_improve}')

        # 检查是否应该早停
        if epochs_no_improve >= patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs!')
            print(f'Best validation loss: {best_val_loss:.4f}')
            break
    
    return train_losses, val_losses

def test_model(model, X_test, y_test, device='cpu', model_path=None):
    """
    测试模型性能并返回评估指标
    
    参数:
        model: 训练好的模型
        X_test: 测试集特征 (numpy数组或类似结构)
        y_test: 测试集真实值 (numpy数组或类似结构)
        device: 运行设备 ('cpu' 或 'cuda')
        model_path: 可选，模型权重文件路径
        
    返回:
        metrics: 包含各项评估指标的字典
        predictions: 模型预测结果
    """
    # 如果有提供模型路径，则加载模型权重
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    
    # 确保模型在正确的设备上
    model.to(device)
    model.eval()
    
    # 转换数据为PyTorch张量并移到设备
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)
    
    # 进行预测
    with torch.no_grad():
        predictions = model(X_test_tensor)
    
    # 将结果移回CPU并转换为numpy数组
    predictions = predictions.cpu().numpy().flatten()
    y_true = y_test_tensor.cpu().numpy().flatten()
    
    # 计算评估指标
    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)
    
    # 打印结果
    print("\n" + "="*50)
    print("模型测试结果 (Boston房价数据集):")
    print(f"- MSE: {mse:.4f}")
    print(f"- RMSE: {rmse:.4f}")
    print(f"- MAE: {mae:.4f}")
    print(f"- R² Score: {r2:.4f}")
    print("="*50 + "\n")
    
    # 返回指标和预测结果
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    return metrics, predictions

##############################################################################
# 实验流程代码
##############################################################################

# 自动下载数据集（约50KB）
boston = fetch_openml(name="boston", version=1, as_frame=True)
df = boston.frame

# 检查数据集中的数据类型
print("数据集特征类型：")
print(df.dtypes)

X = df.drop(columns=['MEDV'])  # 特征
y = df['MEDV']  # 目标

# 确保所有特征都是数值类型
X = X.apply(pd.to_numeric, errors='coerce')  # 将非数值类型转换为数值类型，无法转换的设为 NaN

# 按照 5:1 的比例划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X.values.tolist(), y.values.tolist(), test_size=0.1667, random_state=42)

model = FNN(input_size=13, hidden_size=32, output_size=1, num_hidden_layers=2, activate_function="ReLU")

train_loss, val_loss = train_model(
    model,
    X_train, 
    y_train,
    num_epochs=2000,
    lr=0.001,
    batch_size=32,
    device='mps',
    val_ratio=0.2
)

# 绘制损失曲线
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

test_metrics, preds = test_model(
    model, X_test, y_test,
    device='mps',
    model_path='best_model.pth'
)