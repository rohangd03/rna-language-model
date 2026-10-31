import pickle
import torch
from scripts.probe_bpp import read_recs
from rnalm.targets import dense_targets, pair_mask

cache = pickle.load(open("data/bpp_cache.pkl", "rb"))
for split in ["probe_test", "novel_test"]:
    recs = read_recs("data/splits.csv", split)[:500]   # same subset as the probe
    pos = tot = 0
    for i in range(0, len(recs), 16):
        b = recs[i:i + 16]
        L = max(len(r["seq"]) for r in b)
        lens = torch.tensor([len(r["seq"]) for r in b])
        mask = torch.arange(L)[None] < lens[:, None]
        valid = pair_mask(mask)
        Y = dense_targets([r["id"] for r in b], cache, L)
        pos += int((Y[valid] > 0.5).sum())
        tot += int(valid.sum())
    print(f"{split}: positives {pos}, valid pairs {tot}, base rate {pos / tot:.4f}")
