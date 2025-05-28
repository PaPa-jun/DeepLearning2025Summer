import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import load_data, train_epoch, evaluate
from modules import TextDataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification

texts, labels = load_data("train.jsonl", True)

X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, shuffle=True)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

train_set = TextDataset(X_train, y_train, tokenizer)
val_set = TextDataset(X_val, y_val, tokenizer)

train_loader = DataLoader(train_set, batch_size=32)
val_loader = DataLoader(val_set, batch_size=32)

for epoch in range(3):  # 训练 3 轮
    train_loss = train_epoch(model, train_loader, optimizer, device)
    acc, precision, recall, f1 = evaluate(model, val_loader, device)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Acc: {acc:.4f} | F1: {f1:.4f}")

model.save_pretrained("./best_model")
tokenizer.save_pretrained("./best_model")