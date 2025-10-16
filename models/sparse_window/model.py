import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalWindowAttention(nn.Module):
    """
    Implements local windowed multi-head self-attention for transformers.
    Each query position attends only to a window of nearby key positions.
    """
    def __init__(self, d_model: int, nhead: int, window_size: int):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.window_size = window_size
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        returns: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        qkv = self.qkv_proj(x).reshape(batch_size, seq_len, 3, self.nhead, self.d_model // self.nhead)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]  # (batch, seq_len, nhead, head_dim)

        # Rearrangement for multi-head attention shapes
        q = q.permute(0,2,1,3)
        k = k.permute(0,2,1,3)
        v = v.permute(0,2,1,3)

        attn_outputs = []
        for t in range(seq_len):
            left = max(0, t-self.window_size)
            right = min(seq_len, t+self.window_size+1)
            q_t = q[:,:,t:t+1,:]        # (batch, nhead, 1, head_dim)
            k_local = k[:,:,left:right,:]
            v_local = v[:,:,left:right,:]
            attn_scores = torch.matmul(q_t, k_local.transpose(-2,-1)) / (self.d_model ** 0.5)  # (batch, nhead, 1, window)
            attn_weights = F.softmax(attn_scores, dim=-1)
            out = torch.matmul(attn_weights, v_local)      # (batch, nhead, 1, head_dim)
            attn_outputs.append(out.squeeze(2)) # (batch, nhead, head_dim)
        attn_outputs = torch.stack(attn_outputs, dim=2)   # (batch, nhead, seq_len, head_dim)
        attn_outputs = attn_outputs.permute(0,2,1,3).reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_outputs)

class SparseWindowTransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with local window attention and feedforward network.
    """
    def __init__(self, d_model=128, nhead=4, window_size=2, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.lw_attn = LocalWindowAttention(d_model, nhead, window_size)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.lw_attn(x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class SparseWindowTransformer(nn.Module):
    """
    Transformer stack using local window attention.
    """
    def __init__(self, vocab_size, d_model=128, nhead=4, window_size=2, num_layers=2, dim_feedforward=512, max_seq_len=64, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.layers = nn.ModuleList([
            SparseWindowTransformerEncoderLayer(
                d_model=d_model, nhead=nhead, window_size=window_size,
                dim_feedforward=dim_feedforward, dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.to_logits = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x) + self.pos_embedding[:, :x.size(1), :]
        out = emb
        for layer in self.layers:
            out = layer(out)
        logits = self.to_logits(out)
        return logits

def build_sparse_window_transformer(
    vocab_size, d_model=128, nhead=4, window_size=2, num_layers=2,
    dim_feedforward=512, max_seq_len=64, dropout=0.1
):
    return SparseWindowTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        window_size=window_size,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )

# Optional: basic test function to check construction
def _test():
    vocab_size, seq_len = 100, 32
    model = build_sparse_window_transformer(vocab_size, window_size=2, max_seq_len=seq_len)
    dummy_input = torch.randint(1, vocab_size, (8, seq_len))  # batch x seq_len
    with torch.no_grad():
        output = model(dummy_input)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    _test()
