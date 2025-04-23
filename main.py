import torch
from utlis import load_data, train_model, test_model
from modules import SpamClassifier

device = 'cuda'

train_loader, val_loader, test_loader, tokenizer = load_data("enron_spam_data.csv")

model = SpamClassifier(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=128,
    hidden_size=64,
    output_size=1,
)

criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=10
)

test_model(
    model=model,
    test_loader=test_loader,
    device=device
)