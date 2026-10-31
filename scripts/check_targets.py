import torch
from rnalm.targets import fold_targets, dense_targets, pair_mask

cache = {"t": fold_targets("GGGAAACCC")}
Y = dense_targets(["t"], cache, L=9)
valid = pair_mask(torch.ones(1, 9, dtype=torch.bool))

assert Y[0, 0, 8] > 0.5 and Y[0, 8, 0] == Y[0, 0, 8]   # symmetric fill
assert valid[0, 0, 8]                                   # long-range pair is allowed
assert not valid[0, 0, 2]                               # j - i = 2 is excluded
assert not valid[0, 8, 0]                               # lower triangle is excluded
print("target mapping checks passed.")
