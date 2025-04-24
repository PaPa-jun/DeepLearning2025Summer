import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from modules import SpamClassifierRNN, SpamClassifierAttention, SpamDataset, Tokenizer
from utlis import load_data, train_model, test_model
from sklearn.model_selection import train_test_split

def rnn_experiment(
        texts: list, labels: list, tokenizer: Tokenizer, batch_size: int = 32,
        encoding_length: int = 256, embedding_dim: int = 128, lr: float = 0.001,
        epochs: int = 10, hidden_size: int = 64, output_size: int = 2, device: str = "cpu"
):
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)
    train_dataset = SpamDataset(
        train_texts, train_labels, tokenizer,
        encoding_length=encoding_length,
        padding_side="left",
        truncation_side="right",
        mask=False
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataset = SpamDataset(
        eval_texts, eval_labels, tokenizer,
        encoding_length=encoding_length,
        padding_side="left",
        truncation_side="right",
        mask=False
    )
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
    test_dataset = SpamDataset(
        test_texts, test_labels, tokenizer,
        encoding_length=encoding_length,
        padding_side="left",
        truncation_side="right",
        mask=False
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = SpamClassifierRNN(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        output_size=output_size
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() if output_size == 2 else nn.BCEWithLogitsLoss()
    train_model(
        model=model,
        epochs=epochs,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        attention=False,
        verbose=True,
        plot=True
    )
    metrics = test_model(model, test_loader, criterion, device, attention=False)
    print("-" * 50)
    print("RNN Model Evaluation")
    print("-" * 50)
    print(f"Test Loss: {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("-" * 50)

def attention_experiment(
        texts: list, labels: list, tokenizer: Tokenizer, batch_size: int = 32,
        encoding_length: int = 256, embedding_dim: int = 128, lr: float = 0.001,
        epochs: int = 10, hidden_size: int = 64, output_size: int = 2, num_heads: int = 4,
        dropout: float = 0, rotary: bool = False, use_bias: bool = False, device: str = "cpu"
):
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)
    train_dataset = SpamDataset(
        train_texts, train_labels, tokenizer,
        encoding_length=encoding_length,
        padding_side="left",
        truncation_side="right",
        mask=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataset = SpamDataset(
        eval_texts, eval_labels, tokenizer,
        encoding_length=encoding_length,
        padding_side="left",
        truncation_side="right",
        mask=True
    )
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
    test_dataset = SpamDataset(
        test_texts, test_labels, tokenizer,
        encoding_length=encoding_length,
        padding_side="left",
        truncation_side="right",
        mask=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = SpamClassifierAttention(
        vocab_size=tokenizer.vocab_size,
        encoding_length=encoding_length,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size * num_heads,
        num_heads=num_heads,
        dropout=dropout,
        use_bias=use_bias,
        output_size=output_size,
        rotary=rotary
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() if output_size == 2 else nn.BCEWithLogitsLoss()
    train_model(
        model=model,
        epochs=epochs,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        attention=True,
        verbose=True,
        plot=True
    )
    metrics = test_model(model, test_loader, criterion, device, attention=True)
    print("-" * 50)
    print("Attention Model Evaluation")
    print("-" * 50)
    print(f"Test Loss: {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Spam Classifier Experiment Script")
    
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset CSV file.")
    parser.add_argument("--model_type", type=str, choices=["rnn", "attention"], required=True, help="Type of model to use (RNN or Attention).")

    parser.add_argument("--tokenizer_mode", type=str, default="word", choices=["word", "char"], help="Tokenizer mode (word or char).")
    parser.add_argument("--min_freq", type=int, default=10, help="Minimum frequency for tokenization.")
    parser.add_argument("--special_tokens", type=list, default=["<pad>"], help="List of special tokens for the tokenizer.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on (e.g., 'cpu', 'cuda', 'mps').")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation.")
    parser.add_argument("--encoding_length", type=int, default=256, help="Maximum length of encoded sequences.")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Dimension of word embeddings.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate for the Attention model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size of the model layers.")
    parser.add_argument("--output_size", type=int, default=2, help="Output size of the model (e.g., 2 for binary classification).")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads (only for Attention model).")
    parser.add_argument("--use_bias", action="store_true", help="Whether to use bias in the Attention model.")
    parser.add_argument("--rotary", action="store_true", help="Whether to use rotary embeddings in the Attention model.")

    args = parser.parse_args()

    texts, labels = load_data(args.data_path)
    tokenizer = Tokenizer(texts, args.tokenizer_mode, args.min_freq, special_tokens=args.special_tokens)

    if args.model_type == "rnn":
        rnn_experiment(
            texts=texts,
            labels=labels,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            encoding_length=args.encoding_length,
            embedding_dim=args.embedding_dim,
            lr=args.lr,
            epochs=args.epochs,
            hidden_size=args.hidden_size,
            output_size=args.output_size,
            device=args.device
        )
    elif args.model_type == "attention":
        attention_experiment(
            texts=texts,
            labels=labels,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            encoding_length=args.encoding_length,
            embedding_dim=args.embedding_dim,
            lr=args.lr,
            epochs=args.epochs,
            hidden_size=args.hidden_size,
            output_size=args.output_size,
            num_heads=args.num_heads,
            dropout=args.dropout,
            use_bias=args.use_bias,
            device=args.device,
            rotary=args.rotary
        )

if __name__ == "__main__":
    main()