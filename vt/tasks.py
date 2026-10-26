"""Synthetic algorithmic tasks with a precisely stated ground truth.

Inputs are one-hot sequences of shape (L, V). Train/test splits are disjoint
at the level of distinct strings, so test accuracy measures generalisation.
"""
import numpy as np
import torch


def _is_balanced(seq):
    depth = 0
    for t in seq:
        depth += 1 if t == 0 else -1
        if depth < 0:
            return False
    return depth == 0


def dyck(n, length=16, seed=0):
    """Dyck-1: is a sequence of '(' (0) and ')' (1) balanced?

    Half the samples are balanced. Negatives are hard: half are permutations
    with equal counts whose prefix depth goes negative (so counting alone does
    not solve the task), half are balanced strings with one token flipped.
    """
    if length % 2:
        raise ValueError("Dyck strings need an even length")
    rng = np.random.default_rng(seed)
    base = np.array([0] * (length // 2) + [1] * (length // 2))
    seqs, labels = [], []
    while len(seqs) < n:
        s = rng.permutation(base)
        bal = _is_balanced(s)
        want_pos = len(seqs) % 2 == 0
        if want_pos and bal:
            seqs.append(s); labels.append(1)
        elif not want_pos and not bal:
            if rng.random() < 0.5:
                seqs.append(s); labels.append(0)
            else:
                # flip one token of a balanced string
                while not _is_balanced(s):
                    s = rng.permutation(base)
                s = s.copy()
                i = rng.integers(length)
                s[i] = 1 - s[i]
                seqs.append(s); labels.append(0)
    return np.stack(seqs), np.array(labels), 2


TASKS = {"dyck": dyck}


def make_splits(task, n, length, seed=0, test_frac=0.2):
    seqs, labels, vocab = TASKS[task](n, length, seed)
    # dedupe, then split on a hash of the string so no test string is in train
    uniq = {}
    for s, y in zip(seqs, labels):
        uniq[s.tobytes()] = (s, y)
    keys = sorted(uniq)
    rng = np.random.default_rng(seed + 1)
    is_test = rng.random(len(keys)) < test_frac

    def pack(mask):
        S = np.stack([uniq[k][0] for k, m in zip(keys, mask) if m])
        Y = np.array([uniq[k][1] for k, m in zip(keys, mask) if m])
        # class-balance by subsampling every class to the rarest one's count
        n_min = np.bincount(Y).min()
        keep = np.concatenate([rng.permutation(np.flatnonzero(Y == c))[:n_min]
                               for c in np.unique(Y)])
        keep = rng.permutation(keep)
        S, Y = S[keep], Y[keep]
        X = np.eye(vocab, dtype=np.float32)[S]
        return torch.from_numpy(X), torch.from_numpy(Y).long()

    n_classes = int(labels.max()) + 1
    return pack(~is_test), pack(is_test), vocab, n_classes
