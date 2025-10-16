import torch
import torch.nn as nn
from typing import Optional

class VanillaTransformer(nn.Module):
    """
    Baseline Transformer Encoder model with embedding, positional encoding, and output head.
    Suitable for language/sequence modeling or benchmarking against custom attention variants.
    """
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 512,
                 max_seq_len: int = 64, dropout: float = 0.1, pad_token_id: Optional[int] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.to_logits = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) integer token ids
            src_key_padding_mask: (batch_size, seq_len) mask for pad tokens (optional)
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        # Sanity checks
        assert x.ndim == 2, "Expected input shape (batch_size, seq_len)"
        emb = self.embedding(x) + self.pos_embedding[:, :x.size(1), :]
        out = self.transformer(emb, src_key_padding_mask=src_key_padding_mask)
        logits = self.to_logits(out)
        return logits

def build_vanilla_transformer(
    vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=512, max_seq_len=64, dropout=0.1
):
    return VanillaTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )

# Example test function
def _test():
    vocab_size, seq_len = 100, 32
    model = build_vanilla_transformer(vocab_size, max_seq_len=seq_len)
    dummy_input = torch.randint(1, vocab_size, (8, seq_len))  # batch x seq_len
    with torch.no_grad():
        output = model(dummy_input)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    _test()
