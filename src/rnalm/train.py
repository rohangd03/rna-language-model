"""MLM training loop. First check: the model must overfit one small batch."""
import torch
import torch.nn.functional as F

from rnalm.model import RNAModel, mask_tokens


def mlm_loss(model, input_ids, attention_mask, generator=None):
    """Mask the batch, run the model, and return the cross-entropy over masked positions."""
    masked, labels = mask_tokens(input_ids, attention_mask, generator)
    logits, _ = model(masked, attention_mask)              # (B, L, V)
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),                  # (B*L, V)
        labels.view(-1),                                   # (B*L,)
        ignore_index=-100,                                 # unmasked positions contribute nothing
    )


def make_batch(batch_size=16, seq_len=40):
    """Repeating GGGAAACCC motif with a random phase per row, so the task is learnable."""
    motif = torch.tensor([5, 5, 5, 3, 3, 3, 4, 4, 4])  # G G G A A A C C C
    tiled = motif.repeat(seq_len // len(motif) + 3)
    ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for b in range(batch_size):
        off = torch.randint(0, len(motif), (1,)).item()
        ids[b, 0] = 1                                      # CLS
        ids[b, 1:] = tiled[off:off + seq_len - 1]
    mask = torch.ones_like(ids)                            # no padding in this batch
    return ids, mask


def main():
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device="cpu").manual_seed(0)
    ids, mask = make_batch()
    ids, mask = ids.to(dev), mask.to(dev)

    model = RNAModel(vocab_size=8, d_model=64, n_layers=2, n_heads=4,
                     d_ff=128, max_len=512, dropout=0.0).to(dev)
    print(f"using device: {dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    model.train()
    for step in range(300):
        loss = mlm_loss(model, ids, mask, gen)

        opt.zero_grad()                                         # clear gradients from last step
        loss.backward()                                         # compute new gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # guard against spikes
        opt.step()                                              # update weights

        losses.append(loss.item())
        if step % 50 == 0:
            print(f"step {step:4d}  loss {loss.item():.4f}")

    assert all(torch.isfinite(torch.tensor(losses))), "loss became NaN or inf"
    start, end = sum(losses[:10]) / 10, sum(losses[-10:]) / 10
    print(f"mean loss first 10 steps {start:.4f}, last 10 steps {end:.4f}")
    assert end < 0.5 * start, "loss did not drop enough: check the training loop"
    print("Training overfit check passed.")


if __name__ == "__main__":
    main()