import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from modules import MultiHeadAttention, Tokenizer, SpamDataset
from utlis import load_data

texts, labels = load_data("enron_spam_data.csv")

tokenizer = Tokenizer(texts, "word", 10, ["<pad>"])

dataset = SpamDataset(texts, labels, tokenizer, 4, mask=True)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)   

embedding = nn.Embedding(tokenizer.vocab_size, 128)

mha = MultiHeadAttention(128, 128, 128, 256, 4)


for inputs, labels, masks in dataloader:
    inputs = embedding(inputs)
    out, attention_weights = mha(inputs, inputs, inputs, masks)
    print(attention_weights.shape)
    break

def plot_attention(attention, head_idx):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention[head_idx].detach().cpu().numpy(), cmap="viridis")
    plt.title(f"Head {head_idx + 1}")
    plt.xlabel("Key Positions")
    plt.ylabel("Query Positions")
    plt.show()

batch_index = 0  # 选择一个批次中的样本进行可视化

# 提取当前批次中的注意力权重
attention_weights_sample = attention_weights[batch_index].detach().cpu().numpy()

# 创建子图来显示每个注意力头的权重
num_heads = attention_weights_sample.shape[0]  # 注意力头数量
fig, axes = plt.subplots(1, num_heads, figsize=(20, 5))

# 遍历每个注意力头并绘制热力图
for i in range(num_heads):
    sns.heatmap(attention_weights_sample[i], ax=axes[i], cmap="viridis", cbar=True)
    axes[i].set_title(f"Head {i + 1}")
    axes[i].set_xlabel("Keys")
    axes[i].set_ylabel("Queries")

plt.tight_layout()
plt.show()