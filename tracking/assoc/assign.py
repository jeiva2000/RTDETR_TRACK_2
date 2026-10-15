"""Asignación global track->detección con Hungarian sobre log-probabilidades (inferencia y scheduled sampling).
Cada track puede elegir una detección (exclusiva) o su propio token "no aparece"."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

BIG = 1e6


def hungarian_assign(logits, min_prob=0.0):
    """logits (N,K+1) -> idx (N,) long con el índice de detección o -1 (no aparece), y prob del elegido (N,)."""
    N, K1 = logits.shape
    K = K1 - 1
    if N == 0:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0)
    logp = torch.log_softmax(logits.detach().float(), 1).cpu()
    prob = logp.exp()
    cost = np.full((N, K + N), BIG, dtype=np.float64)
    lp = logp[:, :K].numpy()
    cost[:, :K] = np.where(np.isfinite(lp), -lp, BIG)
    cost[np.arange(N), K + np.arange(N)] = -logp[:, K].numpy()
    r, c = linear_sum_assignment(cost)
    idx = torch.full((N,), -1, dtype=torch.long)
    for i, j in zip(r, c):
        if j < K and cost[i, j] < BIG and prob[i, j] >= min_prob:
            idx[i] = int(j)
    chosen = torch.where(idx >= 0, prob[torch.arange(N), idx.clamp(min=0)], torch.zeros(N))
    return idx, chosen
