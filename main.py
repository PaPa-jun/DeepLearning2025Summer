import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from modules import SpamClassifierRNN, SpamClassifierAttention, SpamDataset, Tokenizer
from utlis import load_data, train_model, test_model
from sklearn.model_selection import train_test_split

def rnn_expiriment(
        texts: list, labels: list, tokenizer: Tokenizer, batch_size: int = 32,
        encoding_length: int = 256, embedding_dim: int = 128, lr: float = 0.001,
        epochs: int = 10, hidden_size: int = 64, output_size: int = 2, device: str = "cpu"
):
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)
    train_dataset = SpamDataset(
        train_texts, train_labels, tokenizer,
        encoding_length = encoding_length,
        padding_side = "left",
        truncation_side= "right",
        mask = False
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataset = SpamDataset(
        eval_texts, eval_labels, tokenizer,
        encoding_length = encoding_length,
        padding_side = "left",
        truncation_side= "right",
        mask = False
    )
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
    test_dataset = SpamDataset(
        test_texts, test_labels, tokenizer,
        encoding_length = encoding_length,
        padding_side = "left",
        truncation_side= "right",
        mask = False
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = SpamClassifierRNN(
        vocab_size = tokenizer.vocab_size,
        embedding_dim = embedding_dim,
        hidden_size = hidden_size,
        output_size = output_size
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() if output_size == 2 else nn.BCEWithLogitsLoss()
    train_model(
        model = model,
        epochs = epochs,
        train_loader = train_loader,
        eval_loader = eval_loader,
        optimizer = optimizer,
        criterion = criterion,
        device = device,
        attention = False,
        verbose = True,
        plot = True
    )
    test_loss, test_acc = test_model(model, test_loader, criterion, device, attention = False)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

def attention_expiriment(
        texts: list, labels: list, tokenizer: Tokenizer, batch_size: int = 32,
        encoding_length: int = 256, embedding_dim: int = 128, lr: float = 0.001,
        epochs: int = 10, hidden_size: int = 64, output_size: int = 2,  num_heads: int = 4,
        use_bias: bool = False, device: str = "cpu"
):
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)
    train_dataset = SpamDataset(
        train_texts, train_labels, tokenizer,
        encoding_length = encoding_length,
        padding_side = "left",
        truncation_side= "right",
        mask = True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataset = SpamDataset(
        eval_texts, eval_labels, tokenizer,
        encoding_length = encoding_length,
        padding_side = "left",
        truncation_side= "right",
        mask = True
    )
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
    test_dataset = SpamDataset(
        test_texts, test_labels, tokenizer,
        encoding_length = encoding_length,
        padding_side = "left",
        truncation_side= "right",
        mask = True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = SpamClassifierAttention(
        vocab_size = tokenizer.vocab_size,
        embedding_dim = embedding_dim,
        hidden_size = hidden_size * num_heads,
        num_heads = num_heads,
        use_bias = use_bias,
        output_size = output_size
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() if output_size == 2 else nn.BCEWithLogitsLoss()
    train_model(
        model = model,
        epochs = epochs,
        train_loader = train_loader,
        eval_loader = eval_loader,
        optimizer = optimizer,
        criterion = criterion,
        device = device,
        attention = True,
        verbose = True,
        plot = True
    )
    test_loss, test_acc = test_model(model, test_loader, criterion, device, attention = True)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

def main():
    texts, labels = load_data("enron_spam_data.csv")
    tokenizer = Tokenizer(texts, "word", 10, special_tokens = ["<pad>"])
    # rnn_expiriment(texts, labels, tokenizer)
    attention_expiriment(texts, labels, tokenizer)

if __name__ == "__main__":
    main()