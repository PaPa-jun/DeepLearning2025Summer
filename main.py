import torch, torch.nn as nn
from modules import SpamClassifierRNN, SpamClassifierAttention, SpamDataset
from utlis import load_data, train_model


def main():
    texts, labels = load_data("enron_spam_data.csv")