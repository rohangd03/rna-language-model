"""Shuffled-target control: score each test sequence with a trained probe, then compare
against its own ViennaRNA targets and against the targets of a random same-length partner."""

import argparse
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from rnalm.model import RNAModel
from rnalm.targets import dense_targets, pair_mask
from scripts.probe_bpp import PairHead2D, residue_features, read_recs


@torch.no_grad()
def predict(model, head, seq, dev):
    feats, mask = residue_features(model, [seq], dev)
    valid = pair_mask(mask)[0].cpu()
    probs = torch.sigmoid(head(feats)[0]).cpu()
    return probs[valid].numpy()


def labels(target_id, cache, L, valid):
    Y = dense_targets([target_id], cache, L)[0]
    return (Y[valid] > 0.5).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/pretrained.pt")
    ap.add_argument("--name", default="pretrained_final")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = RNAModel()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.to(dev).eval()
    head = PairHead2D(model.embedding.d_model).to(dev)
    head.load_state_dict(torch.load(f"models/{args.name}_head.pt", map_location=dev))
    head.eval()

    cache = pickle.load(open("data/bpp_cache.pkl", "rb"))
    rng = random.Random(args.seed)

    for split in ["probe_test", "novel_test"]:
        full = read_recs("data/splits.csv", split)
        by_len = defaultdict(list)
        for r in full:
            by_len[len(r["seq"])].append(r)

        y_real, y_shuf, p_all = [], [], []
        for r in full[: args.n]:
            pool = [q for q in by_len[len(r["seq"])] if q["id"] != r["id"]]
            if not pool:
                continue
            partner = rng.choice(pool)
            L = len(r["seq"])
            valid = pair_mask(torch.ones(1, L, dtype=torch.bool))[0]
            p = predict(model, head, r["seq"], dev)
            y_real.append(labels(r["id"], cache, L, valid))
            y_shuf.append(labels(partner["id"], cache, L, valid))
            p_all.append(p)

        yr, ys = np.concatenate(y_real), np.concatenate(y_shuf)
        p = np.concatenate(p_all)
        print(f"{split}: {len(y_real)} sequences evaluated")
        print(f"  own targets:      AUPRC {average_precision_score(yr, p):.4f}  "
              f"AUROC {roc_auc_score(yr, p):.4f}")
        print(f"  shuffled targets: AUPRC {average_precision_score(ys, p):.4f}  "
              f"AUROC {roc_auc_score(ys, p):.4f}  base rate {ys.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()