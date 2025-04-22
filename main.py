import torch.nn as nn
import pandas as pd

def load_data(data_path: str, split_rate: float = 0.8):
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