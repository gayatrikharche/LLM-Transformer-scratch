import torch.nn as nn
import torch.nn.functional as F
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Head(nn.Module):
    def __init__(self, masked, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.masked = masked
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x)
        q = self.query(x)
        wei = (q @ k.transpose(-2, -1)) * (k.size()[-1]**-0.5)
        if self.masked:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        # wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out, wei

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, n_embd, block_size, masked, dropout):
        super().__init__()
        hidden_dim = n_embd // num_heads
        self.heads = nn.ModuleList([Head(masked, hidden_dim, n_embd, block_size, dropout) for i in range(num_heads)])
        self.proj = nn.Linear(hidden_dim*num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = [h(x) for h in self.heads]
        out2 = torch.cat([h[0] for h in out], dim=-1)
        x=self.proj(out2)
        x = self.dropout(x)
        return x, [h[1] for h in out]

class FeedForward1(nn.Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block1(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd, num_heads, masked, dropout, block_size):
        super().__init__()
        self.sa = MultiHeadAttention(num_heads, n_embd, block_size, masked, dropout)
        self.ffwd = FeedForward1(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, inp):
        x, _ = inp
        sa, attn_maps = self.sa(self.ln1(x))
        x = x + sa
        x = x + self.ffwd(self.ln2(x))
        return x, attn_maps


class Encoder(nn.Module):
    def __init__(self, vocab_size, n_embd, num_layers, num_heads, block_size, dropout, hidden_dimension, clf_dim):
        super(Encoder, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block1(n_embd, num_heads, False, dropout, block_size) for _ in range(num_layers)])
        self.lm_head = nn.Linear(n_embd, hidden_dimension, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.clf=nn.Linear(hidden_dimension, clf_dim, bias=False)

    def forward(self, x):
        B, T = x.size()
        
        tok_emb = self.token_embedding_table(x)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 

        y = tok_emb + pos_emb
        z, attn_maps= self.blocks((y, torch.ones(1, 2, 2).to(device)))
        z = torch.mean(z, dim=1)
        
        logits=self.relu(self.lm_head(z))
        logits = self.clf(logits)  

        return logits, attn_maps
