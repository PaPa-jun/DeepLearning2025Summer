import pandas as pd
import torch.nn as nn, torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Union, List, Tuple, Dict, Optional

def load_data(data_path: str) -> Tuple[List[str], List[str]]:
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
    labels = data_frame['label'].to_list()
    return texts, labels

def train_epoch(
        model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
        criterion: Union[nn.CrossEntropyLoss, nn.BCEWithLogitsLoss],
        device: str = "cpu", attention: bool = False
) -> Tuple[float, float]:
    train_loss = 0
    train_correct = 0
    train_samples = 0
    for stuffs in train_loader:
        if attention:
            inputs, labels, masks = stuffs
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device) \
                if isinstance(criterion, nn.BCEWithLogitsLoss) else labels.long().to(device)
            masks = masks.to(device)
        else:
            inputs, labels = stuffs
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device) \
                if isinstance(criterion, nn.BCEWithLogitsLoss) else labels.long().to(device)

        optimizer.zero_grad()
        predictions = model(inputs, masks) if attention else model(inputs)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += ((predictions > 0.5).float() == labels).sum().item() \
            if isinstance(criterion, nn.BCEWithLogitsLoss) else (torch.argmax(predictions, 1) == labels).sum().item()
        train_samples += labels.shape[0]

    avg_loss = train_loss / len(train_loader)
    accuracy = train_correct / train_samples
    return avg_loss, accuracy

def evaluate_epoch(
        model: nn.Module, eval_loader: DataLoader,
        criterion: Union[nn.CrossEntropyLoss, nn.BCEWithLogitsLoss],
        device: str = "cpu", attention: bool = False
) -> Tuple[float, float]:
    eval_correct = 0
    eval_loss = 0
    eval_samples = 0
    for stuffs in eval_loader:
        if attention:
            inputs, labels, masks = stuffs
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device) \
                if isinstance(criterion, nn.BCEWithLogitsLoss) else labels.long().to(device)
            masks = masks.to(device)
        else:
            inputs, labels = stuffs
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device) \
                if isinstance(criterion, nn.BCEWithLogitsLoss) else labels.long().to(device)
            
        predictions = model(inputs, masks) if attention else model(inputs)
        loss = criterion(predictions, labels)
        
        eval_loss += loss.item()
        eval_correct += ((predictions > 0.5).float() == labels).sum().item() \
            if isinstance(criterion, nn.BCEWithLogitsLoss) else (torch.argmax(predictions, 1) == labels).sum().item()
        eval_samples += labels.shape[0]

    avg_loss = eval_loss / len(eval_loader)
    accuracy = eval_correct / eval_samples
    return avg_loss, accuracy

def train_model(
        model: nn.Module, 
        epochs: int, 
        train_loader: DataLoader, 
        eval_loader: DataLoader,
        optimizer: torch.optim.Optimizer, 
        criterion: Union[nn.CrossEntropyLoss, nn.BCEWithLogitsLoss],
        device: str = "cpu", 
        attention: bool = False, 
        verbose: bool = True, 
        plot: bool = False,
        save: bool = False,
        save_path: Optional[str] = None,
        early_stop: bool = False,
        patience: int = 5, 
        min_delta: float = 0.001
) -> Dict[str, List[float]]:
    assert not save or (save and save_path), "save_path must be provided when save=True"
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "eval_loss": [],
        "eval_acc": []
    }
    
    best_eval_loss = float('inf')
    patience_counter = 0
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, attention
        )
        
        model.eval()
        eval_loss, eval_acc = evaluate_epoch(
            model, eval_loader, criterion, device, attention
        )
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["eval_loss"].append(eval_loss)
        history["eval_acc"].append(eval_acc)

        if eval_loss < best_eval_loss - min_delta:
            best_eval_loss = eval_loss
            patience_counter = 0
            if save:
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience and early_stop:
                if verbose: print(f"Early stopping at epoch {epoch+1} (no improvement >{min_delta})")
                break

        if verbose:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                f"Eval Loss: {eval_loss:.4f} | Eval Acc: {eval_acc:.2%}"
            )
    
    # 绘制损失曲线
    if plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["eval_loss"], label="Eval Loss")
        plt.title("Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history["train_acc"], label="Train Accuracy")
        plt.plot(history["eval_acc"], label="Eval Accuracy")
        plt.title("Accuracy Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return history