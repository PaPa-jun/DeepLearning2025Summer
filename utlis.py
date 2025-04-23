import pandas as pd
import torch.nn as nn, torch
from torch.utils.data import DataLoader
from modules import Tokenizer, SpamDataset
from sklearn.model_selection import train_test_split
from typing import Union, List, Tuple

def load_data(
        data_path: str, split_rate = 0.2, batch_size: int = 32, 
        tokenizer_mode: str = 'char', special_tokens: list = None,
        min_feq: int = 0, encoding_length: int = 256
) -> Tuple[DataLoader, DataLoader, DataLoader, Tokenizer]:
    data_frame = pd.read_csv(data_path).drop(columns=['Date']).rename(
        columns={
            'Message ID': 'id',
            'Subject': 'abstract',
            'Message': 'content',
            'Spam/Ham': 'label',
        }
    ).set_index('id')
    data_frame.dropna(how='any', inplace=True)
    data_frame['label'] = data_frame['label'].map({'spam': 1, 'ham': 0})

    texts = (data_frame["abstract"] + " " + data_frame["content"]).to_list()
    lambels = data_frame['label'].to_list()

    train_val_features, test_features, train_val_labels, test_labels = train_test_split(
        texts, lambels, test_size=split_rate, random_state=42
    )

    train_features, val_features, train_labels, val_labels = train_test_split(
        train_val_features, train_val_labels, test_size=split_rate, random_state=42
    )

    tokenizer = Tokenizer(train_features, tokenizer_mode, min_feq, special_tokens)
    train_set = SpamDataset(train_features, train_labels, tokenizer, encoding_length)
    val_set = SpamDataset(val_features, val_labels, tokenizer, encoding_length)
    test_set = SpamDataset(test_features, test_labels, tokenizer, encoding_length)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader, test_loader, tokenizer

def val_model(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: str = 'cpu') -> float:
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

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

    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total

    return avg_val_loss, val_acc

def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: nn.Module, device: str = 'cpu') -> float:
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicts = (outputs > 0.5).float()
        train_correct += (predicts == labels).sum().item()
        train_total += labels.size(0)

    avg_train_loss = train_loss / len(train_loader)
    train_acc = train_correct / train_total

    return avg_train_loss, train_acc

def plot_loss(train_losses: List[float], val_losses: List[float],
              title: str = 'Loss Plot', xlabel: str = 'Epochs', ylabel: str = 'Loss') -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                criterion: nn.Module, optimizer: nn.Module, epochs: int = 10, 
                device: str = 'cpu', save_path: str = 'best_model.pth', save: bool = False) -> None:
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    model.to(device)

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = val_model(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if  val_loss < best_val_loss:
            best_val_loss = val_loss
            if save:
                torch.save(model.state_dict(), save_path)
                print(f'Model saved to {save_path}')

        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    plot_loss(train_losses, val_losses)
    print(f'Training complete. Best validation loss: {best_val_loss:.4f}')

def test_model(model: nn.Module, test_loader: DataLoader, device: str = 'cpu') -> float:
    model.to(device)
    accs = []
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(inputs)
        predicts = (outputs > 0.5).float()
        test_correct = (predicts == labels).sum().item()
        test_total = labels.size(0)
        test_acc = test_correct / test_total
        accs.append(test_acc)
    avg_test_acc = sum(accs) / len(accs)
    print(f'Test Accuracy: {avg_test_acc:.4f}')
    return avg_test_acc