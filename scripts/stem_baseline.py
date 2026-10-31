"""Rule-based baseline: score a pair 1 if it is canonical AND its diagonal
neighbour (i+1, j-1) is also canonical (a stem-like pattern), else 0.
Evaluated on the same pairs and sequences as probe_bpp.py."""

import csv
import pickle

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from rnalm.targets import dense_targets, pair_mask
from rnalm.tokenizer import encode

# Token IDs: A=3, C=4, G=5, U=6
CANON = [(3, 6), (6, 3), (4, 5), (5, 4), (5, 6), (6, 5)]   # AU, UA, CG, GC, GU, UG
CANMAT = torch.zeros(8, 8, dtype=torch.bool)
for a, b in CANON:
    CANMAT[a, b] = True


def read_recs(path, split):
    with open(path, newline="") as fh:
        return [{"id": r["id"], "seq": r["seq"]}
                for r in csv.DictReader(fh) if r["split"] == split]


def stem_score(ids):
    """ids: (L,) residue token IDs with CLS removed. Returns (L, L) scores."""
    can = CANMAT[ids[:, None], ids[None, :]].float()        # canonical[i, j]
    L = ids.numel()
    s = torch.zeros(L, L)
    if L > 1:
        s[:-1, 1:] = can[:-1, 1:] * can[1:, :-1]             # (i+1, j-1) also canonical
    return s * can


def main():
    cache = pickle.load(open("data/bpp_cache.pkl", "rb"))
    for split in ["probe_test", "novel_test"]:
        recs = read_recs("data/splits.csv", split)[:500]    # same subset as the probe
        ys, ps = [], []
        for r in recs:
            ids = torch.tensor(encode(r["seq"])[1:], dtype=torch.long)   # drop CLS
            L = ids.numel()
            scores = stem_score(ids)
            valid = pair_mask(torch.ones(1, L, dtype=torch.bool))[0]
            Y = dense_targets([r["id"]], cache, L)[0]
            ys.append((Y[valid] > 0.5).numpy())
            ps.append(scores[valid].numpy())
        y, p = np.concatenate(ys), np.concatenate(ps)
        print(f"{split}: base rate {y.mean():.4f}  "
              f"stem-rule AUPRC {average_precision_score(y, p):.4f}  "
              f"AUROC {roc_auc_score(y, p):.4f}", flush=True)


if __name__ == "__main__":
    main()