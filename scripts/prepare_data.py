"""Build the cleaned, deduplicated, split-labelled dataset from a family-labelled Stockholm file.

Example:
  python scripts/prepare_data.py --stockholm data/Rfam.seed
  mmseqs easy-cluster data/all.fa data/clu data/tmp --min-seq-id 0.8 -c 0.8
  python scripts/prepare_data.py --stockholm data/Rfam.seed --clusters data/clu_cluster.tsv
"""

import argparse
from collections import Counter
from pathlib import Path

from rnalm.data import (
    parse_stockholm, clean_records, write_fasta, load_clusters,
    assign_splits, write_splits,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stockholm", required=True)
    ap.add_argument("--clusters", default=None, help="mmseqs easy-cluster *_cluster.tsv")
    ap.add_argument("--out", default="data/splits.csv")
    ap.add_argument("--fasta_out", default="data/all.fa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--novel_frac", type=float, default=0.10)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    recs = clean_records(parse_stockholm(args.stockholm))
    write_fasta(recs, args.fasta_out)
    print(f"{len(recs)} sequences after cleaning/dedup, "
          f"{len({r['family'] for r in recs})} families")

    clusters = load_clusters(args.clusters) if args.clusters else {}
    if not clusters:
        print("WARNING: no cluster file given. Near-duplicates may leak across splits; "
              "run the mmseqs step in the README and re-run with --clusters.")

    recs = assign_splits(recs, clusters, seed=args.seed, novel_frac=args.novel_frac)
    write_splits(recs, args.out)

    counts = Counter(r["split"] for r in recs)
    for s in ("pretrain", "probe_train", "probe_test", "novel_test"):
        fams = len({r["family"] for r in recs if r["split"] == s})
        print(f"{s:12s} seqs={counts[s]:7d} families={fams}")


if __name__ == "__main__":
    main()
