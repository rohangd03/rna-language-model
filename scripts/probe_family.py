"""Probe: classify Rfam family from pooled embeddings with a linear (logistic) probe.

Trained and tested on probe_train / probe_test (families seen in pretraining, sequences
split by cluster). Families with fewer than --min_train probe-train sequences are dropped.

Examples:
  python scripts/probe_family.py --encoder pretrained --model_dir models/pretrained --name pretrained
  python scripts/probe_family.py --encoder random     --model_dir models/random     --name random
  python scripts/probe_family.py --encoder onehot                                    --name onehot
  python scripts/probe_family.py --encoder kmer3                                     --name kmer3
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from rnalm.data import read_split
from rnalm.embed import Encoder


def kmer_features(seqs, k=3):
    idx = {"A": 0, "C": 1, "G": 2, "U": 3}
    X = np.zeros((len(seqs), 4 ** k), dtype=np.float32)
    for b, s in enumerate(seqs):
        for i in range(len(s) - k + 1):
            kmer = s[i:i + k]
            if all(c in idx for c in kmer):
                code = 0
                for c in kmer:
                    code = code * 4 + idx[c]
                X[b, code] += 1
        tot = X[b].sum()
        if tot:
            X[b] /= tot
    return X


def pooled_features(enc, seqs, bs=64):
    out = []
    for i in range(0, len(seqs), bs):
        f, m = enc(seqs[i:i + bs])
        m = m.float()[..., None]
        out.append(((f * m).sum(1) / m.sum(1).clamp(min=1)).numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True,
                    choices=["pretrained", "random", "onehot", "kmer3"])
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default="data/splits.csv")
    ap.add_argument("--min_train", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train = read_split(args.data, "probe_train")
    test = read_split(args.data, "probe_test")
    counts = Counter(r["family"] for r in train)
    keep = {f for f, c in counts.items() if c >= args.min_train}
    train = [r for r in train if r["family"] in keep]
    test = [r for r in test if r["family"] in keep]
    fams = sorted(keep)
    fid = {f: i for i, f in enumerate(fams)}
    ytr = np.array([fid[r["family"]] for r in train])
    yte = np.array([fid[r["family"]] for r in test])
    seqs_tr = [r["seq"] for r in train]
    seqs_te = [r["seq"] for r in test]

    if args.encoder == "kmer3":
        Xtr, Xte = kmer_features(seqs_tr), kmer_features(seqs_te)
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        enc = Encoder(args.encoder, args.model_dir, dev)
        Xtr, Xte = pooled_features(enc, seqs_tr), pooled_features(enc, seqs_te)

    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=args.seed)
    clf.fit(scaler.transform(Xtr), ytr)
    pred = clf.predict(scaler.transform(Xte))

    majority = Counter(ytr).most_common(1)[0][0]
    result = {
        "name": args.name,
        "encoder": args.encoder,
        "probe": "rfam_family_classification",
        "n_families": len(fams),
        "n_train": len(ytr),
        "n_test": len(yte),
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro")),
        "majority_baseline_accuracy": float(np.mean(yte == majority)),
    }
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{args.name}_family.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
