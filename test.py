from modules import Tokenizer, SpamDataset
from torch.utils.data import DataLoader
import pandas as pd

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

print(tokenizer.decode(tokenizer.encode("HELLO"), skip_special_tokens=False))