# RNA language model: pretraining and probing

A ~5M-parameter BERT-style masked language model trained on Rfam sequences, with its
embeddings probed for (1) base-pair probabilities from ViennaRNA and (2) Rfam family
classification. Compared against a random-init model with identical architecture,
a one-hot baseline, a 3-mer baseline, and (optionally) RNA-FM.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Data

Download Rfam seed alignments (family-labelled, Stockholm format):

```bash
mkdir -p data && cd data
wget https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.seed.gz && gunzip Rfam.seed.gz
cd ..
```

Build the dataset, then cluster to remove near-duplicates and rebuild splits:

```bash
python scripts/prepare_data.py --stockholm data/Rfam.seed          # writes data/all.fa
mmseqs easy-cluster data/all.fa data/clu data/tmp --min-seq-id 0.8 -c 0.8
python scripts/prepare_data.py --stockholm data/Rfam.seed --clusters data/clu_cluster.tsv
```

(MMseqs2 is installable via conda or from the GitHub release binaries.)

## Split protocol

Three questions need three different splits, and mixing them up is the most common way
these results end up inflated.

- **Novel families (`novel_test`, ~10% of families).** Never seen in pretraining or probe
  training. Used to test whether the embedding captures structure that transfers to
  sequences with no family-level neighbours.
- **Sequence clusters (`pretrain` / `probe_train` / `probe_test`).** Within the seen
  families, MMseqs2 clusters at 80% identity are assigned whole to one split, so a
  probe test sequence never has a near-duplicate in pretraining.
- **Family classification** is a supervised task over seen families, so it uses
  `probe_train` vs `probe_test`. It cannot be evaluated on novel families by design.

Pretraining sees only `pretrain`. The 2% MLM validation split is drawn from `pretrain`.

## Run order

```bash
# 1. Pretrain (or save the random-init control)
python scripts/pretrain.py --out models/pretrained --epochs 30
python scripts/pretrain.py --out models/random --random_init

# 2. Base-pair probe (ViennaRNA targets are computed once and cached)
python scripts/probe_bpp.py --encoder onehot                                  --name onehot
python scripts/probe_bpp.py --encoder random   --model_dir models/random      --name random
python scripts/probe_bpp.py --encoder pretrained --model_dir models/pretrained --name pretrained

# 3. Family probe
python scripts/probe_family.py --encoder kmer3                                 --name kmer3
python scripts/probe_family.py --encoder onehot                                --name onehot
python scripts/probe_family.py --encoder random   --model_dir models/random    --name random
python scripts/probe_family.py --encoder pretrained --model_dir models/pretrained --name pretrained
```

Results land in `results/*.json`. Each BPP result reports both `test_seen_families` and
`test_novel_families`; the gap between them is the most informative number in this project.
