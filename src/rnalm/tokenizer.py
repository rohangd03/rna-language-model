import torch

"""One-token-per-nucleotide tokenizer for RNA."""

PAD, CLS, MASK = "[PAD]", "[CLS]", "[MASK]"
VOCAB = [PAD, CLS, MASK, "A", "C", "G", "U", "N"]
NUCS = ["A", "C", "G", "U", "N"]

TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}
ID_TO_TOKEN = {i: tok for tok, i in TOKEN_TO_ID.items()}

PAD_ID = TOKEN_TO_ID[PAD]    # 0
CLS_ID = TOKEN_TO_ID[CLS]    # 1
MASK_ID = TOKEN_TO_ID[MASK]  # 2
MAX_LEN = 512


def clean_seq(seq: str) -> str:
    """Return an uppercase RNA string containing only A, C, G, U, N."""
    seq = seq.upper().replace("T", "U")
    return "".join(c for c in seq if c in "ACGUN")


def encode(seq: str) -> list[int]:
    """Turn an RNA string into token IDs, with [CLS] at index 0."""
    seq = clean_seq(seq)
    ids = [CLS_ID]
    for c in seq:
        ids.append(TOKEN_TO_ID[c])
    return ids


def decode(ids: list[int]) -> str:
    """Turn token IDs back into a string, skipping PAD and CLS."""
    return "".join(ID_TO_TOKEN[i] for i in ids if i not in (PAD_ID, CLS_ID))


def pad_batch(id_lists: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:

    if not id_lists:
        return torch.empty((0, 0), dtype=torch.long), torch.empty((0, 0), dtype=torch.long)

    batch_size = len(id_lists)
    max_seq_len = max(len(seq) for seq in id_lists)
    input_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)

    for i, seq in enumerate(id_lists):
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, :len(seq)] = 1

    return input_ids, attention_mask 



if __name__ == "__main__":
    ids = encode("GGGAAACCC")
    assert len(ids) == 10
    assert ids[0] == CLS_ID
    assert ids[1] == TOKEN_TO_ID["G"]
    assert ids == [1, 5, 5, 5, 3, 3, 3, 4, 4, 4]
    assert decode(ids) == "GGGAAACCC"

    # T becomes U, lowercase is accepted, gaps are dropped
    assert encode("acgt") == [1, 3, 4, 5, 6]
    assert encode("GG-AA.") == encode("GGAA")

    print("all tokenizer checks passed")
