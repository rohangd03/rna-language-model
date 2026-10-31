import numpy as np
import RNA


def fold_targets(seq):
    """Return sparse base-pair probabilities as (i, j, p), 0-indexed, with i < j."""
    fc = RNA.fold_compound(seq)
    fc.pf()                                  # McCaskill partition function
    raw = np.array(fc.bpp(), dtype=np.float64)  # (n+1) x (n+1), 1-indexed
    upper = np.triu(raw, k=1)[1:, 1:]        # keep i < j, drop the unused row/column 0
    ii, jj = np.nonzero(upper > 0.01)        # sparsify: most entries are ~0
    return ii, jj, upper[ii, jj]


if __name__ == "__main__":
    ii, jj, pp = fold_targets("GGGAAACCC")
    pairs = {(int(i), int(j)): round(float(p), 3) for i, j, p in zip(ii, jj, pp)}
    print(pairs)
    assert pairs[(0, 8)] > 0.5    # outer G pairs with outer C (positions 1 and 9)
    assert pairs[(2, 6)] > 0.5    # inner G pairs with inner C (positions 3 and 7)
    assert all(i < j for i, j in pairs)
    assert all(0 <= p <= 1 for p in pairs.values())
    print("fold_targets checks passed.")