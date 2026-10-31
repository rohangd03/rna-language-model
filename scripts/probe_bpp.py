"""Base-pair probe: frozen encoder hidden states -> pair logits, trained on ViennaRNA targets."""
import argparse
import csv
import json
import pickle
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from rnalm.model import RNAModel
from rnalm.targets import dense_targets, fold_targets, pair_mask
from rnalm.tokenizer import encode, pad_batch


def read_recs(path, split):
    with open(path, newline="") as fh:
        return [{"id": r["id"], "seq": r["seq"]}
                for r in csv.DictReader(fh) if r["split"] == split]


def build_cache(recs, path):
    """Fold every sequence once with ViennaRNA and store the sparse targets on disk."""
    p = Path(path)
    cache = pickle.load(open(p, "rb")) if p.exists() else {}
    todo = [r for r in recs if r["id"] not in cache]
    if todo:
        print(f"folding {len(todo)} sequences with ViennaRNA...")
        with Pool() as pool:
            results = pool.imap(fold_targets, [r["seq"] for r in todo], chunksize=32)
            for r, targets in zip(todo, results):
                cache[r["id"]] = targets
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            pickle.dump(cache, fh)
    return cache


def load_encoder(checkpoint, dev):
    model = RNAModel()
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(dev).eval()


@torch.no_grad()
def residue_features(model, seqs, dev):
    """Hidden states per residue, with CLS removed so index i is residue i."""
    ids, attn = pad_batch([encode(s) for s in seqs])
    _, hidden = model(ids.to(dev), attn.to(dev))
    return hidden[:, 1:], attn[:, 1:].bool()


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

@torch.no_grad()
def evaluate(model, head, recs, cache, dev, bs=16, max_n=500):
    head.eval()
    recs = recs[:max_n]
    ys, ps, loss_sum, n_pairs = [], [], 0.0, 0
    for i in range(0, len(recs), bs):
        batch = recs[i:i + bs]
        feats, mask = residue_features(model, [r["seq"] for r in batch], dev)
        logits = head(feats)
        valid = pair_mask(mask)
        Y = dense_targets([r["id"] for r in batch], cache, mask.size(1), dev)
        loss_sum += F.binary_cross_entropy_with_logits(
            logits[valid], Y[valid], reduction="sum").item()
        n_pairs += int(valid.sum())
        ys.append(Y[valid].cpu().numpy())
        ps.append(torch.sigmoid(logits[valid]).cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return {
        "loss": loss_sum / max(n_pairs, 1),
        "auprc": float(average_precision_score(y > 0.5, p)),
        "auroc": float(roc_auc_score(y > 0.5, p)),
        "n_seqs": len(recs),
        "n_pairs": n_pairs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/pretrained.pt",
                    help="state_dict path, or 'none' for an untrained model")
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default="data/splits.csv")
    ap.add_argument("--cache", default="data/bpp_cache.pkl")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_eval", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    train = read_recs(args.data, "probe_train")
    rng.shuffle(train)
    n_val = len(train) // 10
    val, train = train[:n_val], train[n_val:]
    test_seen = read_recs(args.data, "probe_test")
    test_novel = read_recs(args.data, "novel_test")

    cache = build_cache(train + val + test_seen + test_novel, args.cache)

    checkpoint = None if args.checkpoint == "none" else args.checkpoint
    model = load_encoder(checkpoint, dev)
    head = PairHead2D(model.embedding.d_model).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val, best_state = float("inf"), None
    for epoch in range(args.epochs):
        head.train()
        rng.shuffle(train)
        for i in range(0, len(train), args.bs):
            batch = train[i:i + args.bs]
            feats, mask = residue_features(model, [r["seq"] for r in batch], dev)
            logits = head(feats)
            valid = pair_mask(mask)
            Y = dense_targets([r["id"] for r in batch], cache, mask.size(1), dev)
            loss = F.binary_cross_entropy_with_logits(logits[valid], Y[valid])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (i // args.bs) % 200 == 0:
                print(f"  Epoch {epoch} batch {i // args.bs}  loss {loss.item():.4f}", flush=True)

        v = evaluate(model, head, val, cache, dev)
        print(f"epoch {epoch:3d}  val_loss {v['loss']:.4f}  val_auprc {v['auprc']:.4f}")
        if v["loss"] < best_val:
            best_val = v["loss"]
            best_state = {k: t.detach().clone() for k, t in head.state_dict().items()}

    head.load_state_dict(best_state) # type: ignore
    torch.save(head.state_dict(), f"models/{args.name}_head.pt")

    result = {
        "name": args.name,
        "checkpoint": args.checkpoint,
        "val_loss": best_val,
        "test_seen_families": evaluate(model, head, test_seen, cache, dev, max_n=args.max_eval),
        "test_novel_families": evaluate(model, head, test_novel, cache, dev, max_n=args.max_eval),
    }
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{args.name}_bpp.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
