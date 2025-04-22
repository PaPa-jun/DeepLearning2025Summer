from modules import Tokenizer, SpamDataset, SpamClassifier
from torch.utils.data import DataLoader
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def losses_plot(epochs, train_losses, val_losses):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss', marker='x')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.show()

def train(model, lr, epochs, train_loader, val_loader, verbose=True, device="cpu", save_path="best_model.pth", save=False):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.to(device)
    best_val_loss = float('inf')

    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicts = (outputs > 0.5).float()
            train_correct += (predicts == labels).sum().item()
            train_total += labels.size(0)
        
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicts = (outputs > 0.5).float()
                val_correct += (predicts == labels).sum().item()
                val_total += labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if verbose:
            print(f'Epoch [{epoch+1}/{epochs}]: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}. Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        best_val_loss = avg_val_loss if avg_val_loss < best_val_loss else best_val_loss
        if save:
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at epoch {epoch+1} with Val Loss: {avg_val_loss:.4f}")

    losses_plot(epochs, train_losses, val_losses)

    print("训练完成！最佳模型已保存为:", save_path)

data_frame = pd.read_csv('enron_spam_data.csv').drop(columns=['Date']).rename(
    columns={
        'Message ID': 'id',
        'Subject': 'abstract',
        'Message': 'content',
        'Spam/Ham': 'label',
    }
).set_index('id')
data_frame.dropna(how='any', inplace=True)
data_frame['label'] = data_frame['label'].map({'spam': 1, 'ham': 0})

texts = (data_frame['abstract'] + ' ' + data_frame["content"]).to_list()
lables = data_frame["label"].to_list()

tokenizer = Tokenizer(texts, mode='char')

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, lables, test_size=0.2, random_state=42)
train_dataset = SpamDataset(train_texts, train_labels, tokenizer)
val_dataset = SpamDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = SpamClassifier(tokenizer.vocab_size, embedding_dim=128, hidden_size=64, output_size=1)
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

train(model, lr=0.001, epochs=10, train_loader=train_loader, val_loader=val_loader, device=device, save_path="best_model.pth", save=True)