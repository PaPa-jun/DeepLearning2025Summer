import pandas as pd
import torch

from utlis import load_data
from modules import SpamDataset, Tokenizer

data_frame = pd.read_csv("enron_spam_data.csv").drop(columns=['Date']).rename(
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

test_texts = [
    "this is a test sentences ."
]

tokenizer = Tokenizer(texts, "word", 10, ["<pad>"])

trainset = SpamDataset(test_texts, labels, tokenizer, mask=True)

for x, y, m in trainset:
    print(f'{x}\n{y}\n{m}')
    break