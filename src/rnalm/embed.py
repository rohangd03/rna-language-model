"""Uniform interface over the encoders being compared."""
import torch
from transformers import BertModel

from .tokenizer import encode, pad_batch, NUCS


class Encoder:
    """Call with a list of cleaned sequences; returns (features (B,L,D), mask (B,L)).

    Feature index i corresponds to residue i (CLS is dropped), so features line up
    with ViennaRNA's 1-indexed base-pair matrix after subtracting one.
    kind: 'pretrained' | 'random' (both load a BERT checkpoint dir) | 'onehot'.
    """

    def __init__(self, kind, model_dir=None, device="cpu"):
        self.kind, self.device = kind, device
        if kind in ("pretrained", "random"):
            assert model_dir, "model_dir required for transformer encoders"
            self.model = BertModel.from_pretrained(model_dir, add_pooling_layer=False)
            self.model.to(device).eval()
            self.dim = self.model.config.hidden_size
        elif kind == "onehot":
            self.dim = 4
        else:
            raise ValueError(f"unknown encoder kind: {kind}")

    @torch.no_grad()
    def __call__(self, seqs):
        lens = [len(s) for s in seqs]
        if self.kind == "onehot":
            L = max(lens)
            feats = torch.zeros(len(seqs), L, 4)
            for b, s in enumerate(seqs):
                for i, c in enumerate(s):
                    if c in NUCS:
                        feats[b, i, NUCS.index(c)] = 1.0
        else:
            ids, attn = pad_batch([encode(s) for s in seqs])
            hid = self.model(
                input_ids=ids.to(self.device), attention_mask=attn.to(self.device)
            ).last_hidden_state
            feats = hid[:, 1:].cpu()
        L = feats.shape[1]
        mask = torch.arange(L)[None, :] < torch.tensor(lens)[:, None]
        return feats.float(), mask
