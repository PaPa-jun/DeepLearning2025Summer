import pandas as pd
import torch, torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.nn import Parameter
from torch.functional import F

from sklearn.model_selection import train_test_split

from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.processors import TemplateProcessing

from transformers import AutoTokenizer

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

texts = data_frame["content"].to_list()

class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return (
            encoding["input_ids"].squeeze(),
            # encoding["attention_mask"].squeeze(),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )
    
# tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))

# tokenizer.normalizer = normalizers.Sequence([
#     normalizers.NFKC(),
#     normalizers.Lowercase()
# ])

# special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]"]
# trainer = WordLevelTrainer(
#     vocab_size=30000,
#     min_frequency=20,
#     special_tokens=special_tokens
# )

# tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

# tokenizer.train_from_iterator(texts, trainer=trainer)
# tokenizer.post_processor = TemplateProcessing(
#     single="[CLS] $A [SEP]",
#     pair="[CLS] $A [SEP] $B:1 [SEP]:1",
#     special_tokens=[
#         ("[CLS]", tokenizer.token_to_id("[CLS]")),
#         ("[SEP]", tokenizer.token_to_id("[SEP]")),
#     ],
# )

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class BetterRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=2, 
                         bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)
    
def train(model, lr, epochs, train_loader, val_loader, verbose=True, device="cpu", save_path="best_model.pth"):
    criterion = nn.BCEWithLogitsLoss()
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
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            if verbose:
                print(f"Best model saved at epoch {epoch+1} with Val Loss: {avg_val_loss:.4f}")

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

    print("训练完成！最佳模型已保存为:", save_path)

# Constract Training data.
labels = data_frame["label"].to_list()

train_mail, val_mail, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

train_dataset = SpamDataset(train_mail, train_labels, tokenizer)
val_dataset = SpamDataset(val_mail, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BetterRNN(
    vocab_size=tokenizer.vocab_size,
    embed_dim=128,
    hidden_dim=64,
    output_dim=1
).to(device)
train(model, 0.001, 20, train_loader, val_loader, device="cuda")