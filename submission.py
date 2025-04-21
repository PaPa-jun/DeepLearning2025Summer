import pandas as pd, re
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# Load Dataset
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

# Tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(
    vocab_size=5000,
    min_frequency=2,
    special_tokens=["[UNK]", "[PAD]"]
)

corups ='. ' .join(data_frame["content"].to_list() + data_frame["abstract"].to_list())
with open('corpus.txt', 'w') as file:
    file.write(corups)
    
tokenizer.train(files=['corpus.txt'], trainer=trainer)
tokenizer.save("tokenizer.json")

# Model
