"""Paso 7: asignación con fila y columna dustbin.
Matriz aumentada (D+1, T+1): la fila D es el dustbin de las detecciones ("track nuevo": una detección asignada a la columna dustbin) y la columna T el de los
tracks ("no observado en t": un track asignado a la fila dustbin); todas sus entradas valen z, un score aprendible. Marginales como en SuperGlue (cada detección y
cada track tienen masa 1; el dustbin de detecciones, la masa T válidos; el de tracks, D válidas), todo normalizado por (D+T).
- Entrenamiento: Sinkhorn en log-dominio (iters iteraciones) -> log P (B,D+1,T+1) con cada fila de detección y cada columna de track sumando 1; L_assign es la NLL contra la asignación GT.
- Inferencia: Hungarian sobre s (con temperatura) y un umbral mínimo sobre la PROBABILIDAD de la pareja, P_ij = exp(log P_ij) de Sinkhorn (h3.min_prob, 0.5 por defecto): una pareja
  solo se acepta si P_ij >= min_prob. P está siempre en [0, 1] y se interpreta igual en cualquier modelo (la escala de s no: la fija el entrenamiento); con 0.5 cada fila y cada columna
  tienen a lo sumo una pareja aceptada, así que el resultado es siempre una asignación exclusiva.
Con relleno y máscaras (det_valid, trk_valid): las filas/columnas inválidas no reciben masa (NEG) y los resultados de las válidas no dependen del relleno; D = 0 y T = 0 son válidos."""
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

NEG = -1e4


def log_sinkhorn(scores, z, det_valid, trk_valid, iters=20):
    """scores (B,D,T), z escalar, det_valid (B,D), trk_valid (B,T) -> log P (B,D+1,T+1)."""
    B, D, T = scores.shape
    one = torch.ones(B, 1, dtype=torch.bool, device=scores.device)
    Z = torch.cat([torch.cat([scores, z.expand(B, D, 1)], 2), z.expand(B, 1, T + 1)], 1)
    vr, vc = torch.cat([det_valid, one], 1), torch.cat([trk_valid, one], 1)
    Z = Z.masked_fill(~(vr[:, :, None] & vc[:, None, :]), NEG)
    m, n = det_valid.sum(1).float(), trk_valid.sum(1).float()
    norm = -torch.log((m + n).clamp(min=1))                                                      # (B,)
    neg = torch.full_like(norm, NEG)
    log_mu = torch.cat([torch.where(det_valid, norm[:, None], NEG), torch.where(n > 0, torch.log(n.clamp(min=1)) + norm, neg)[:, None]], 1)
    log_nu = torch.cat([torch.where(trk_valid, norm[:, None], NEG), torch.where(m > 0, torch.log(m.clamp(min=1)) + norm, neg)[:, None]], 1)
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v[:, None, :], dim=2)
        v = log_nu - torch.logsumexp(Z + u[:, :, None], dim=1)
    return Z + u[:, :, None] + v[:, None, :] - norm[:, None, None]


def sinkhorn_nll(logP, target_det, target_trk, det_valid, trk_valid):
    """NLL contra la asignación GT, dustbin incluido. target_det (B,D): columna del track GT (0..T-1) o T (dustbin = track nuevo / falso positivo);
    target_trk (B,T): detección GT del track o D (dustbin = no observado). Cada pareja GT cuenta una vez; cada detección sin pareja y cada track sin pareja,
    una. Devuelve (suma, nº de términos); sin términos, un cero conectado al grafo (DDP)."""
    B, D1, T1 = logP.shape
    D, T = D1 - 1, T1 - 1
    lp_det = logP[:, :D, :].gather(2, target_det[..., None]).squeeze(-1) if D > 0 else logP.new_zeros(B, 0)
    lone = trk_valid & (target_trk == D)
    terms = -(lp_det * det_valid).sum() - (logP[:, D, :T] * lone).sum()
    n = det_valid.sum() + lone.sum()
    return (terms, n) if n > 0 else (logP.clamp(min=NEG).sum() * 0.0, n)


class Assignment(nn.Module):
    def __init__(self, iters=20):
        super().__init__()
        self.iters = iters
        self.z = nn.Parameter(torch.zeros(()))                      # score del dustbin (aprendible)

    def forward(self, s, det_valid, trk_valid):
        return log_sinkhorn(s, self.z, det_valid, trk_valid, self.iters)

    @torch.no_grad()
    def decide(self, s, logP, det_valid, trk_valid, target=None, use_model=None, min_prob=0.5):
        """Asignación (B,T) con la detección de cada track o D (= no observado). Donde use_model es True: Hungarian sobre s y se descartan las parejas con
        P_ij < min_prob (P = exp(logP) sin el dustbin); donde es False (y hay target, teacher forcing): el target."""
        B, D, T = s.shape
        out = target.clone() if target is not None else torch.full((B, T), D, dtype=torch.long, device=s.device)
        if use_model is None:
            use_model = torch.ones(B, dtype=torch.bool)
        P = logP[:, :D, :T].exp()
        for b in range(B):
            if not bool(use_model[b]):
                continue
            out[b] = D
            rows, cols = det_valid[b].nonzero().flatten(), trk_valid[b].nonzero().flatten()
            if len(rows) == 0 or len(cols) == 0:
                continue
            r, c = linear_sum_assignment(-s[b][rows][:, cols].float().cpu().numpy())
            for i, j in zip(r, c):
                if float(P[b, rows[i], cols[j]]) >= min_prob:
                    out[b, cols[j]] = rows[i]
        return out
