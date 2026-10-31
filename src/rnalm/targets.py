"""ViennaRNA base-pair targets and the helpers that map them onto residue indices."""
import numpy as np
import torch
import RNA

MIN_LOOP = 4  # pairs with j - i < 4 are excluded (no hairpin is that tight)


def fold_targets(seq):
    """Return sparse base-pair probabilities as (i, j, p), 0-indexed, with i < j."""
    fc = RNA.fold_compound(seq)
    fc.pf()                                     # McCaskill partition function
    raw = np.array(fc.bpp(), dtype=np.float64)  # (n+1) x (n+1), 1-indexed
    upper = np.triu(raw, k=1)[1:, 1:]           # keep i < j, drop row/column 0
    ii, jj = np.nonzero(upper > 0.01)           # sparsify
    return ii, jj, upper[ii, jj]


def dense_targets(ids, cache, L, dev="cpu"):
    """Build a (B, L, L) matrix from sparse entries, filling both (i, j) and (j, i)."""
    Y = torch.zeros(len(ids), L, L)
    for b, rid in enumerate(ids):
        ii, jj, pp = cache[rid]
        ii, jj = torch.as_tensor(ii), torch.as_tensor(jj)
        pp = torch.as_tensor(pp, dtype=torch.float32)
        Y[b, ii, jj] = pp
        Y[b, jj, ii] = pp
    return Y.to(dev)


def pair_mask(mask):
    """(B, L) residue mask -> (B, L, L) mask over real pairs with j - i >= MIN_LOOP."""
    idx = torch.arange(mask.size(1), device=mask.device)
    band = (idx[None, :] - idx[:, None]) >= MIN_LOOP  # band[i, j] = (j - i >= MIN_LOOP)
    return mask[:, :, None] & mask[:, None, :] & band[None]
