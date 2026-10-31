"""Masked-LM pretraining on the 'pretrain' split of data/splits.csv."""
import argparse
import csv
import json
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from rnalm.model import RNAModel, mask_tokens
from rnalm.tokenizer import encode, pad_batch
from tqdm import tqdm


class IdsDataset(Dataset):
    def __init__(self, seqs):
        self.ids = [encode(s) for s in seqs]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i]


def read_seqs(path, split):
    with open(path, newline="") as fh:
        return [r["seq"] for r in csv.DictReader(fh) if r["split"] == split]


def mlm_step(model, masked, attn, labels):
    logits, _ = model(masked, attn)
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
    )

def bucketed_batches(seqs, bs, rng):
    """Shuffle, sort within chunks by length, then cut into batches of similar length."""
    idx = list(range(len(seqs)))
    rng.shuffle(idx)
    chunk = bs * 50
    batches = []
    for i in range(0, len(idx), chunk):
        block = sorted(idx[i:i + chunk], key=lambda j: len(seqs[j]))
        batches += [block[k:k + bs] for k in range(0, len(block), bs)]
    rng.shuffle(batches)
    return batches


@torch.no_grad()
def evaluate(model, loader, gen, dev):
    model.eval()
    total, n = 0.0, 0
    for ids, attn in loader:
        masked, labels = mask_tokens(ids, attn, gen)
        loss = mlm_step(model, masked.to(dev), attn.to(dev), labels.to(dev))
        total += loss.item()
        n += 1
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/splits.csv")
    ap.add_argument("--out", default="models/pretrained.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--limit", type=int, default=0, help="use N sequences only (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    seqs = read_seqs(args.data, "pretrain")
    seqs = [s for s in seqs if len(s) <= 511]
    rng.shuffle(seqs)
    if args.limit:
        seqs = seqs[: args.limit]
    n_val = max(1, int(0.02 * len(seqs)))
    val_seqs, train_seqs = seqs[:n_val], seqs[n_val:]
    print(f"{len(train_seqs)} train / {len(val_seqs)} validation sequences")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNAModel().to(dev)
    print(f"using device: {dev}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    gen_tr = torch.Generator(device="cpu").manual_seed(args.seed)
    gen_va = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    train_dl = DataLoader(IdsDataset(train_seqs), batch_size=args.bs, shuffle=True,
                          collate_fn=lambda b: pad_batch(b))
    val_dl = DataLoader(IdsDataset(val_seqs), batch_size=args.bs, shuffle=False,
                        collate_fn=lambda b: pad_batch(b))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_dl)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * total_steps), total_steps)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best, log = float("inf"), []

    for epoch in range(args.epochs):
        model.train()
        running, n = 0.0, 0
        for b in bucketed_batches(train_seqs, args.bs, rng):
            ids, attn = pad_batch([encode(train_seqs[j]) for j in b])
            masked, labels = mask_tokens(ids, attn, gen_tr)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                loss = mlm_step(model, masked.to(dev), attn.to(dev), labels.to(dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            n += 1

        val = evaluate(model, val_dl, gen_va, dev)
        torch.save(model.state_dict(), f"models/pretrained_ep{epoch:03d}.pt")
        print(f"epoch {epoch:3d}  train {running/n:.4f}  val {val:.4f}  "
            f"peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)

        log.append({"epoch": epoch, "train_loss": running / n, "val_loss": val})
        if val < best:
            best = val
            torch.save(model.state_dict(), out)

    Path(out.with_suffix(".json")).write_text(json.dumps(log, indent=2))
    print(f"best validation loss {best:.4f}, saved to {out}")


if __name__ == "__main__":
    main()