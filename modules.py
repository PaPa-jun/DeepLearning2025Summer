import torch.nn as nn
import torch, math
from collections import Counter
from torch.utils.data import Dataset
from torch.nn import Parameter
from torch.functional import F
from typing import Union, List, Tuple

class Tokenizer:
    def __init__(self, texts: list, mode: str = 'word', min_freq: int = 0, special_tokens: list = None):
        self.mode = mode
        self.tokens = self._split(texts, mode)
        counter = self._count_corpus()
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.special_tokens = special_tokens if special_tokens else []
        self.padding = False
        self.truncation = False
        
        self.idx2token = ['<unk>'] + self.special_tokens
        self.token2idx = {token: idx for idx, token in enumerate(self.idx2token)}
        
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token2idx:
                self.idx2token.append(token)
                self.token2idx[token] = len(self.idx2token) - 1

    def __len__(self) -> int:
        return len(self.idx2token)

    def __getitem__(self, tokens: Union[list, tuple, str]) -> int | List[int]:
        if not isinstance(tokens, (list, tuple)):
            return self.token2idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def _count_corpus(self) -> Counter:
        if len(self.tokens) == 0 or isinstance(self.tokens[0], list):
            tokens = [token for line in self.tokens for token in line]
        return Counter(tokens)

    def _split(self, texts: list, mode: str) -> list:
        if mode == 'word':
            return [line.split() for line in texts]
        if mode == 'char':
            return [list(line) for line in texts]
        else:
            raise ValueError("Wrong mode: 'word' or 'char'!")
        
    def enable_padding(self, min_length: int, padding_side: str = 'right') -> None:
        if '<pad>' not in self.special_tokens:
            self.special_tokens.append('<pad>')
            self.idx2token.insert(1, '<pad>')
            self.token2idx = {token: idx for idx, token in enumerate(self.idx2token)}
            self.pad_idx = len(self.special_tokens) - 1
        else:
            self.pad_idx = self.token2idx['<pad>']
        self.padding = True
        self.padding_side = padding_side
        self.min_length = min_length

    def enable_truncation(self, max_length: int, truncation_side: str = 'right') -> None:
        self.truncation = True
        self.max_length = max_length
        self.truncation_side = truncation_side

    def encode(self, text: str) -> Tuple[List[int], List[int]]:
        tokens = self._split([text], self.mode)[0]
        ids = [self.token2idx.get(t, self.unk) for t in tokens]

        if self.truncation and len(ids) > self.max_length:
            if self.truncation_side == "left":
                ids = ids[self.max_length :]
            elif self.truncation_side == "right":
                ids = ids[: self.max_length]
            else:
                raise ValueError("'trunction_side' should be either 'right' or 'left'!")
            
        valid_length = len(ids)
        mask = [1] * valid_length
            
        if self.padding and len(ids) < self.min_length:
            if self.padding_side == "left":
                ids = [self.pad_idx] * (self.min_length - valid_length) + ids
                mask = [0] * (self.min_length - valid_length) + [1] * valid_length
            elif self.padding_side == "right":
                ids += [self.pad_idx] * (self.min_length - valid_length)
                mask = [1] * valid_length + [0] * (self.min_length - valid_length)
            else:
                raise ValueError("'padding_side' should be either 'right' or 'left'!")
            
        return ids, mask
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for idx in ids:
            token = self.idx2token[idx]
            if skip_special_tokens and token in {*self.special_tokens, '<unk>'}:
                continue
            tokens.append(token)
        return ' '.join(tokens) if self.mode == 'word' else ''.join(tokens)

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs
    
    @property
    def vocab_size(self):
        return self.__len__()

class SpamDataset(Dataset):
    def __init__(
            self, texts: list, labels: list, tokenizer: Tokenizer, encoding_length: int = 256,
            padding_side: str = 'left', truncation_side: str = "right", mask: bool = False
    ):
        super(SpamDataset, self).__init__()
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.mask = mask

        self.tokenizer.enable_padding(encoding_length, padding_side)
        self.tokenizer.enable_truncation(encoding_length, truncation_side)

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[index]
        ids, mask = self.tokenizer.encode(text)

        x = torch.tensor(ids, dtype=torch.long)
        y = torch.tensor(self.labels[index], dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.int)

        if self.mask:
            return x, y, mask
        else:
            return x, y
    
class DotProductAttention(nn.Module):
    def __init__(self, scale: float = None, dropout: float = 0):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.scale = scale

    def forward(self, queries: torch.Tensor, keys: torch.Tensor,
                values: torch.Tensor, masks: torch.Tensor = None
    ):
        d_k = queries.shape[-1]
        scale = self.scale if self.scale is not None else math.sqrt(d_k)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / scale

        if masks is not None:
            masks = masks.unsqueeze(-2).expand(-1, queries.shape[-2], -1)
            scores = scores.masked_fill((masks == 0), float("-inf"))

        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(attn_weights, values)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(
            self, key_size: int, query_size: int, value_size: int,
            hidden_size: int, num_heads: int, dropout: float = 0,
            use_bias: bool = False
    ):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.num_heads = num_heads
        
        self.W_q = nn.Linear(query_size, hidden_size, bias=use_bias)
        self.W_k = nn.Linear(key_size, hidden_size, bias=use_bias)
        self.W_v = nn.Linear(value_size, hidden_size, bias=use_bias)
        self.W_o = nn.Linear(hidden_size, hidden_size, bias=use_bias)

        self.attention = DotProductAttention(dropout=dropout)

    def _transpose_qkv(self, inputs: torch.Tensor):
        inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], self.num_heads, -1)
        inputs = inputs.permute(0, 2, 1, 3)
        return inputs.reshape(-1, inputs.shape[2], inputs.shape[3])
    
    def _transpose_output(self, inputs: torch.Tensor):
        inputs = inputs.reshape(-1, self.num_heads, inputs.shape[1], inputs.shape[2])
        inputs = inputs.permute(0, 2, 1, 3)
        return inputs.reshape(inputs.shape[0], inputs.shape[1], -1)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, 
                values: torch.Tensor, masks: torch.Tensor = None):
        Q = self._transpose_qkv(self.W_q(queries))
        K = self._transpose_qkv(self.W_k(keys))
        V = self._transpose_qkv(self.W_v(values))

        if masks is not None:
            masks = masks.repeat_interleave(self.num_heads, dim=0)

        attn_output = self.attention(Q, K, V, masks)
        attn_output = self._transpose_output(attn_output)
        output = self.W_o(attn_output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0, max_length=1000):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_length, embedding_dim))
        X = torch.arange(max_length, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, embedding_dim, 2, dtype=torch.float32) / embedding_dim)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X: torch.Tensor):
        X = X * math.sqrt(self.embedding_dim)
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
    
# RNN Model
class SpamClassifierRNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, output_size: int = 1):
        super(SpamClassifierRNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim),
            nn.RNN(embedding_dim, hidden_size, num_layers=1, batch_first=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, inputs):
        _, H = self.encoder(inputs)
        outputs = self.decoder(H[-1])
        return outputs
    
# Attention Model
class SpamClassifierAttention(nn.Module):
    def __init__(
            self, vocab_size: int, embedding_dim: int, hidden_size: int,
            num_heads: int, drop_out: int = 0, use_bias: bool = False,
            output_size: int = 1
    ):
        super(SpamClassifierAttention, self).__init__()
        self.encoder = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim),
            PositionalEncoding(embedding_dim)
        )
        self.attention = MultiHeadAttention(
            embedding_dim, embedding_dim, embedding_dim,
            hidden_size, num_heads, drop_out, use_bias
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, inputs: torch.Tensor, masks: torch.Tensor):
        inputs = self.encoder(inputs)
        attn_out = self.attention(inputs, inputs, inputs, masks)
        attn_out = attn_out.mean(dim=1)
        predictions = self.decoder(attn_out)
        return predictions