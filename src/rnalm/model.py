"""BERT-style masked LM sized to a few million parameters."""
import torch
from transformers import BertConfig, BertForMaskedLM
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

## Create vector embeddings for RNA sequence
class RNAEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int = 8,
        d_model: int = 256,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.token_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=0,
        )
        self.position_embeddings = nn.Embedding(
            num_embeddings=max_len,
            embedding_dim=d_model,
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_len:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds model max_len ({self.max_len})"
            )

        positions = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        tok_emb = self.token_embeddings(input_ids)
        pos_emb = self.position_embeddings(positions)
        x = self.layer_norm(tok_emb + pos_emb)
        return self.dropout(x)





## Multi-head attention with masking 
def scaled_dot_product_attention(q, k, v, key_padding_mask=None):
    """
    q, k, v: (batch, seq_len, d_head)
    key_padding_mask: (batch, seq_len) bool, True for real tokens, False for PAD
    Returns: output (batch, seq_len, d_head), weights (batch, seq_len, seq_len)
    """
    d_head = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_head)  # (B, L, L)

    if key_padding_mask is not None:
        scores = scores.masked_fill(~key_padding_mask[:, None, :], float("-inf"))

    weights = torch.softmax(scores, dim=-1)  # each row sums to 1
    return weights @ v, weights


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(t):
            return t.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if key_padding_mask is not None:
            # Mask out PAD tokens: shape (B, 1, 1, L)
            scores = scores.masked_fill(~key_padding_mask[:, None, None, :], -1e9)

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        ctx = (weights @ v).transpose(1, 2).reshape(B, L, D)
        return self.out(ctx), weights


class FeedForward(nn.Module):
    def __init__(self, d_model: int = 256, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)



# Transformer block
class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block: attention and feed-forward, each with a residual."""

    def __init__(self, d_model: int = 256, n_heads: int = 8, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, key_padding_mask=None):
        attn_out, weights = self.attn(self.norm1(x), key_padding_mask)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, weights



# Full model
class RNAModel(nn.Module):
    """Transformer encoder with a masked-LM head."""

    def __init__(
        self,
        vocab_size: int = 8,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = RNAEmbedding(vocab_size, d_model, max_len, dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        key_padding_mask = attention_mask.bool() if attention_mask is not None else None

        x = self.embedding(input_ids)
        for block in self.blocks:
            x, _ = block(x, key_padding_mask)
        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)
        return logits, hidden


MASK_ID = 2
MLM_PROB = 0.15

def mask_tokens(input_ids, attention_mask, generator=None):
    """
    input_ids:      (batch, seq_len) long, unmasked (any device)
    attention_mask: (batch, seq_len) with 1 for real tokens
    Returns: masked_ids, labels, both on input_ids.device.
    All random draws happen on the CPU because the generator lives there.
    """
    out_device = input_ids.device
    ids = input_ids.detach().to("cpu")
    attn = attention_mask.detach().to("cpu")

    candidates = attn.bool() & (ids >= 3)                  # nucleotides only
    selected = (
        torch.rand(ids.shape, generator=generator, device="cpu") < MLM_PROB
    ) & candidates

    for b in range(ids.size(0)):                           # at least one target per row
        if not selected[b].any() and candidates[b].any():
            pos = candidates[b].nonzero().flatten()
            pick = pos[
                torch.randint(len(pos), (1,), generator=generator, device="cpu")
            ]
            selected[b, pick] = True

    labels = ids.clone()
    labels[~selected] = -100

    masked_ids = ids.clone()
    r = torch.rand(ids.shape, generator=generator, device="cpu")
    masked_ids[selected & (r < 0.8)] = MASK_ID             # 80%: [MASK]
    random_tok = torch.randint(3, 7, ids.shape, generator=generator, device="cpu")
    swap = selected & (r >= 0.8) & (r < 0.9)               # 10%: random nucleotide
    masked_ids[swap] = random_tok[swap]                    # remaining 10%: unchanged

    return masked_ids.to(out_device), labels.to(out_device)

# 2D convolutional layer
class ResBlock2D(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=dilation, dilation=dilation)

    def forward(self, x):
        return x + self.conv(F.gelu(self.norm(x)))


class PairHead2D(nn.Module):
    """Pair logits from residue states, with 2D convolutions over the pair map."""

    def __init__(self, d_model, ch=32, dilations=(1, 2, 4, 8), max_sep=128):
        super().__init__()
        self.proj_i = nn.Linear(d_model, ch)
        self.proj_j = nn.Linear(d_model, ch)
        self.sep_emb = nn.Embedding(max_sep + 1, ch)   # learned bias by sequence separation
        self.max_sep = max_sep
        self.blocks = nn.ModuleList([ResBlock2D(ch, d) for d in dilations])
        self.out = nn.Conv2d(ch, 1, kernel_size=1)

    def forward(self, h):                                   # h: (B, L, D)
        B, L, _ = h.shape
        a, b = self.proj_i(h), self.proj_j(h)
        x = a[:, :, None, :] * b[:, None, :, :] + a[:, :, None, :] + b[:, None, :, :]
        idx = torch.arange(L, device=h.device)
        sep = (idx[None, :] - idx[:, None]).abs().clamp(max=self.max_sep)
        x = (x + self.sep_emb(sep)[None]).permute(0, 3, 1, 2)   # (B, ch, L, L)
        for blk in self.blocks:
            x = blk(x)
        s = self.out(x).squeeze(1)                           # (B, L, L)
        return 0.5 * (s + s.transpose(1, 2))