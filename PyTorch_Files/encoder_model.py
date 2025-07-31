import torch
import torch.nn as nn
import toml
from tokenizers import Tokenizer

# config = toml.load('config.toml')
# tokenizer = Tokenizer.from_file('./BPE/bpe_tokenizer.json')
# vocab_size = tokenizer.get_vocab_size()
# inputs = {'input_ids': torch.tensor([   1,   15, 1075,  126,  194,  430,  105,  873]), 'attention_mask': torch.tensor([1, 1, 1, 1, 1, 1, 1, 1]), 'label': torch.tensor(1)}
# x = inputs['input_ids']
# embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config["model"]["embedding_dim"])
# print(embeddings)
# print(embeddings(x).detach())

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        # d_model = Embedding_dim
        # d_k = d_q = d_model/num_heads -> num_heads is number of attention heads
        super().__init__()
        self.d_k = d_k

        # nn.Linear will project each context window's dimensions from d_model to d_k
        # nn.Linear only operates on the last dimension
        self.W_q = nn.Linear(in_features=d_model, out_features=d_k)      # [128, 16] assuming 8 attention heads
        self.W_k = nn.Linear(in_features=d_model, out_features=d_k)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_v)

    def forward(self, x):
        # x has shape [batch_size, max_len, embedding_dim] i.e. [32, 1024, 128]
        # queries will have shape [32, 1024, 16]
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        # transpose(-2, -1) transposes the last 2 dimensions crucial for correct matrix multiplication
        attn_scores = torch.softmax(queries @ keys.transpose(-2, -1)/self.d_k**0.5, dim=-1) @ values
        return attn_scores

# sa = SelfAttention(d_model=config["model"]["embedding_dim"], d_k=16, d_v=16)
# print(sa(embeddings(x).detach()))

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model, d_k, d_v, n_heads):
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfAttention(d_model=d_model, d_k=d_k, d_v=d_v) for _ in range(n_heads)]
        )
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    
# mhsa = MultiHeadSelfAttention(d_model=config["model"]["embedding_dim"], d_k=16, d_v=16, n_heads=8)

# # Add and Norm
# out = mhsa(embeddings(x).detach())

class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(size))
        self.beta = nn.Parameter(torch.zeros(size))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x-mean) / (std + self.eps) + self.beta
    
# layer_norm = LayerNorm(size=128)
# print(layer_norm(out).mean(), layer_norm(out).std())


class FeedForwardNetwork(nn.Module):
    # d_model = Embedding Dimension
    # d_ff = Hidden dimension
    def __init__(self, d_model, d_ff, dropout=1e-2):
        super().__init__()
        self.fc1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.fc2 = nn.Linear(in_features=d_ff, out_features=d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
# ffn = FeedForwardNetwork(d_model=config["model"]["embedding_dim"], d_ff=config["model"]["hidden_layers"])
# ffn_out = ffn(out)
# print(ffn_out)
# final_out = layer_norm(out + ffn_out)
# print(final_out)

class Encoder(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Multi Head Self Attention + Add and Norm
        x = self.norm1(x + self.mhsa(x))

        # Feed Forward Network + Add and Norm
        x = self.norm2(x + self.ffn(x))
        return x
    
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, d_k, d_v, n_heads, d_ff, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        
        # Run the same Encoder block for each layer
        self.layers = nn.ModuleList([
            Encoder(d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads, d_ff=d_ff) for _ in range(n_layers)
        ])
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return x

class Classifier(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, d_k, d_v, n_heads, d_ff, n_layers, n_classes):
        super().__init__()
        self.encoder = Transformer(vocab_size=vocab_size, d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads, d_ff=d_ff, n_layers=n_layers)
        self.pos_encoding = nn.Embedding(num_embeddings=max_len, embedding_dim=d_model)
        self.classifier = nn.Linear(in_features=d_model, out_features=n_classes)

    def forward(self, input_ids):
        # Positional Encoding Implementation - Simply adding an encoding for each position to the existing embeddings
        positions = torch.arange(start=0, end=input_ids.size(1), device=input_ids.device).unsqueeze(dim=0)
        x = self.encoder.embedding(input_ids) + self.pos_encoding(positions)

        for layer in self.encoder.layers:
            x = layer(x)

        # Classification Head Mean Pooling
        x = x.mean(dim=1)
        return self.classifier(x)