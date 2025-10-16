import torch.nn as nn
import triton
import triton.language as tl
import torch

@triton.jit
def windowed_attn_kernel(
    Q, K, V,  # Input tensors
    O,         # Output tensor
    seq_len,   # Sequence length
    window_size, # Window size
    head_dim: tl.constexpr,  # Head dimension (now a constexpr)
    stride_qz, stride_qs, stride_qh,  # strides for Q
    stride_kz, stride_ks, stride_kh,  # strides for K
    stride_vz, stride_vs, stride_vh,  # strides for V
    stride_oz, stride_os, stride_oh,  # strides for O
):
    # Get program IDs
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    
    # Offsets for Q, K, V, O
    q_offset = tl.program_id(2)
    
    q_ptr = Q + batch_id * stride_qz + head_id * stride_qh + q_offset * stride_qs
    
    # Load query for the current position
    q = tl.load(q_ptr + tl.arange(0, head_dim))
    
    # Initialize output values
    o_val = tl.zeros([head_dim], dtype=tl.float32)
    max_score = -float('inf')
    total_scores = tl.zeros([1], dtype=tl.float32)

    # Loop over the window
    for offset in range(-window_size, window_size + 1):
        j = q_offset + offset
        is_valid = (j >= 0) & (j < seq_len)
        
        # Only load if within window and sequence bounds
        if is_valid:
            k_ptr = K + batch_id * stride_kz + head_id * stride_kh + j * stride_ks
            v_ptr = V + batch_id * stride_vz + head_id * stride_vh + j * stride_vs
            
            k = tl.load(k_ptr + tl.arange(0, head_dim))
            v = tl.load(v_ptr + tl.arange(0, head_dim))
            
            # Compute score and update max
            score = tl.sum(q * k) / tl.sqrt(float(head_dim))
            #score = tl.dot(q, k) / tl.sqrt(float(head_dim))
            #score = tl.dot(q, tl.trans(k)) / tl.sqrt(float(head_dim))
            max_score = tl.maximum(max_score, score)

            # Accumulate scores and values
            o_val += tl.exp(score - max_score) * v
            total_scores += tl.exp(score - max_score)

    # Normalize and write output
    o_ptr = O + batch_id * stride_oz + head_id * stride_oh + q_offset * stride_os
    tl.store(o_ptr + tl.arange(0, head_dim), o_val / total_scores)

def triton_windowed_attention(q, k, v, window_size):
    # Dimensions
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Output tensor
    o = torch.empty_like(q, device=q.device)

    # Get strides
    stride_qz, stride_qs, stride_qh = q.stride(0), q.stride(1), q.stride(2)
    stride_kz, stride_ks, stride_kh = k.stride(0), k.stride(1), k.stride(2)
    stride_vz, stride_vs, stride_vh = v.stride(0), v.stride(1), v.stride(2)
    stride_oz, stride_os, stride_oh = o.stride(0), o.stride(1), o.stride(2)

    # Grid for the kernel
    grid = (batch_size, num_heads, seq_len)
    
    # Kernel call
    windowed_attn_kernel[grid](
        q, k, v, o, seq_len, window_size, head_dim,
        stride_qz, stride_qs, stride_qh,
        stride_kz, stride_ks, stride_kh,
        stride_vz, stride_vs, stride_vh,
        stride_oz, stride_os, stride_oh,
    )
    
    return o

class TritonWindowedSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, window_size):
        super().__init__()
        self.nhead = nhead
        self.window_size = window_size
        self.d_model = d_model
        assert d_model % nhead == 0
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        B, S, D = x.shape
        H = self.nhead
        q = self.q_proj(x).reshape(B, S, H, self.head_dim).permute(0,2,1,3).contiguous()
        k = self.k_proj(x).reshape(B, S, H, self.head_dim).permute(0,2,1,3).contiguous()
        v = self.v_proj(x).reshape(B, S, H, self.head_dim).permute(0,2,1,3).contiguous()
        o = triton_windowed_attention(q, k, v, self.window_size)
        out = o.permute(0,2,1,3).reshape(B, S, D).contiguous()
        return self.out_proj(out)

class TritonSparseWindowTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=128, nhead=4, window_size=2, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.attn = TritonWindowedSelfAttention(d_model, nhead, window_size)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x

class TritonSparseWindowTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, window_size=2, num_layers=2, dim_feedforward=512, max_seq_len=64, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.layers = nn.ModuleList([
            TritonSparseWindowTransformerEncoderLayer(
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

def build_triton_sparse_window_transformer(
    vocab_size, d_model=128, nhead=4, window_size=2, num_layers=2,
    dim_feedforward=512, max_seq_len=64, dropout=0.1
):
    return TritonSparseWindowTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        window_size=window_size,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )
