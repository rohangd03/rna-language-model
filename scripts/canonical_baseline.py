"""Reference: AUPRC of a rule-based score (1 if the pair is Watson-Crick or GU) on the same pairs."""
import pickle

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from rnalm.targets import dense_targets, pair_mask
from rnalm.tokenizer import encode, pad_batch
from scripts.probe_bpp import read_recs

PAIRS = [(3, 6), (6, 3), (4, 5), (5, 4), (5, 6), (6, 5)]  # AU, UA, CG, GC, GU, UG
CAN = torch.zeros(8, 8, dtype=torch.bool)
for a, b in PAIRS:
    CAN[a, b] = True

cache = pickle.load(open("data/bpp_cache.pkl", "rb"))
for split in ["probe_test", "novel_test"]:
    recs = read_recs("data/splits.csv", split)[:500]
    ys, ps = [], []
    for i in range(0, len(recs), 16):
        batch = recs[i:i + 16]
        ids, attn = pad_batch([encode(r["seq"]) for r in batch])
        ids, mask = ids[:, 1:], attn[:, 1:].bool()          # drop CLS, as in the probe
        scores = CAN[ids[:, :, None], ids[:, None, :]].float()
        valid = pair_mask(mask)
        Y = dense_targets([r["id"] for r in batch], cache, mask.size(1))
        ys.append(Y[valid].numpy())
        ps.append(scores[valid].numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    auprc = average_precision_score(y > 0.5, p)
    print(f"{split}: canonical-pair AUPRC {auprc:.4f}")
