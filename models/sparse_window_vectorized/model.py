import torch
import torch.nn as nn

class EfficientWindowAttention(nn.Module):
    def __init__(self, d_model, nhead, window_size):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.window_size = window_size
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.size()
        qkv = self.qkv_proj(x).reshape(batch_size, seq_len, 3, self.nhead, self.d_model // self.nhead)
        # Project and arrange for multihead attention
        q, k, v = [qkv[:,:,i].permute(0,2,1,3) for i in range(3)]
        # (batch, nhead, seq_len, head_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_model ** 0.5) # (batch, nhead, seq_len, seq_len)

        # Create a local window mask (on device)
        mask = torch.zeros(seq_len, seq_len, device=x.device)
        for i in range(seq_len):
            left = max(0, i-self.window_size)
            right = min(seq_len, i+self.window_size+1)
            mask[i, left:right] = 1
        # mask shape (1, 1, seq_len, seq_len) for broadcasting
        mask = mask.unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(mask==0, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1,2).reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(out)

class SparseWindowVectorizedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=128, nhead=4, window_size=2, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.lw_attn = EfficientWindowAttention(d_model, nhead, window_size)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out = self.lw_attn(x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class SparseWindowVectorizedTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, window_size=2, num_layers=2, dim_feedforward=512, max_seq_len=64, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.layers = nn.ModuleList([
            SparseWindowVectorizedTransformerEncoderLayer(
                d_model=d_model, nhead=nhead, window_size=window_size,
                dim_feedforward=dim_feedforward, dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.to_logits = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.embedding(x) + self.pos_embedding[:, :x.size(1), :]
        out = emb
        for layer in self.layers:
            out = layer(out)
        logits = self.to_logits(out)
        return logits

def build_sparse_window_vectorized_transformer(
    vocab_size, d_model=128, nhead=4, window_size=2, num_layers=2, dim_feedforward=512, max_seq_len=64, dropout=0.1
):
    return SparseWindowVectorizedTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        window_size=window_size,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )

