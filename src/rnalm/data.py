"""Parsing, cleaning, deduplication and split assignment for RNA family data.

Split protocol (see README for the reasoning):
  - ~10% of Rfam families are "novel": they never appear in pretraining or probe training.
  - Among the remaining families, sequence clusters (MMseqs2, >=80% identity) are assigned
    as whole units to pretrain / probe_train / probe_test, so near-duplicates never straddle splits.
  - Any cluster touching a novel family is entirely novel.
"""
import csv
import random
from collections import defaultdict

from .tokenizer import clean_seq, MAX_LEN

MIN_LEN = 20


def parse_stockholm(path):
    """Yield (family, seq_name, ungapped_seq) from a Stockholm file such as Rfam.seed."""
    fam, seqs = None, {}
    with open(path, "r", encoding="latin-1") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#=GF AC"):
                fam = line.split()[2].split(".")[0]
                seqs = {}
            elif line.startswith("//"):
                if fam is not None:
                    for name, chunks in seqs.items():
                        yield fam, name, "".join(chunks)
                fam, seqs = None, {}
            elif not line.strip() or line.startswith("#"):
                continue
            else:
                parts = line.split()
                if len(parts) >= 2 and fam is not None:
                    seqs.setdefault(parts[0], []).append(parts[1])


def clean_records(raw):
    """Clean, length-filter, and exact-deduplicate. Returns dicts with id, family, seq."""
    seen, out = set(), []
    for fam, _name, seq in raw:
        s = clean_seq(seq)
        if not (MIN_LEN <= len(s) <= MAX_LEN) or s in seen:
            continue
        seen.add(s)
        out.append({"id": f"{fam}_{len(out)}", "family": fam, "seq": s})
    return out


def write_fasta(recs, path):
    with open(path, "w") as fh:
        for r in recs:
            fh.write(f">{r['id']}\n{r['seq']}\n")


def load_clusters(tsv_path):
    """Map member id -> representative id from `mmseqs easy-cluster` output (rep<TAB>member)."""
    mapping = {}
    with open(tsv_path) as fh:
        for line in fh:
            rep, member = line.rstrip("\n").split("\t")[:2]
            mapping[member] = rep
    return mapping


def assign_splits(recs, clusters, seed=0, novel_frac=0.10, fracs=(0.6, 0.2, 0.2)):
    rng = random.Random(seed)
    families = sorted({r["family"] for r in recs})
    rng.shuffle(families)
    n_novel = max(1, round(novel_frac * len(families)))
    novel_fams = set(families[:n_novel])

    groups = defaultdict(list)
    for r in recs:
        groups[clusters.get(r["id"], r["id"])].append(r)

    novel_keys = {k for k, g in groups.items() if any(r["family"] in novel_fams for r in g)}
    normal_keys = sorted(k for k in groups if k not in novel_keys)
    rng.shuffle(normal_keys)

    n = len(normal_keys)
    c1 = round(fracs[0] * n)
    c2 = c1 + round(fracs[1] * n)
    split_of = {}
    for i, k in enumerate(normal_keys):
        split_of[k] = "pretrain" if i < c1 else ("probe_train" if i < c2 else "probe_test")

    out = []
    for k, g in groups.items():
        s = "novel_test" if k in novel_keys else split_of[k]
        for r in g:
            out.append({**r, "cluster": k, "split": s})
    return out


def write_splits(recs, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "family", "seq", "cluster", "split"])
        w.writeheader()
        w.writerows(recs)


def read_split(path, split):
    with open(path, newline="") as fh:
        return [
            {"id": r["id"], "family": r["family"], "seq": r["seq"]}
            for r in csv.DictReader(fh)
            if r["split"] == split
        ]
