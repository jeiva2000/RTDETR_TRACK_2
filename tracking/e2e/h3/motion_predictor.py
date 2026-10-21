"""Paso 2: predictor de movimiento (Kalman aprendido). GRU sobre el historial de las últimas K cajas de cada track.
Entrada por paso: [cx, cy, w, h, Δcx, Δcy, Δw, Δh] (Δ respecto al paso anterior, escalado por delta_scale para acondicionar).
Salidas por track (B,N): caja predicha para el frame actual, velocidad predicha y embedding de movimiento m (emb_dim).
- La velocidad es el delta de caja POR FRAME en la parametrización de tracking/e2e/motion.py (dx/w, dy/h, log w'/w, log h'/h), y
  box_pred = apply_delta(última caja, velocidad · frames desde la última asignación): así la velocidad queda supervisada por la pérdida de la caja.
- Historial con padding y máscara: hist (B,N,K,4) con la caja MÁS RECIENTE en el índice -1 y hist_len (B,N) estados válidos (contiguos al final).
  Con < 2 estados el Δ vale 0 (no hay paso anterior); con 0 estados (hueco libre) la salida es finita y se enmascara fuera."""
import torch
import torch.nn as nn

from tracking.e2e.heads import mlp
from tracking.e2e.motion import apply_delta

MU_SCALE = 0.1      # salida cruda O(1) -> velocidad O(0.1) por frame (en tamaños de caja)


class MotionPredictor(nn.Module):
    def __init__(self, gru_dim=128, emb_dim=128, hidden=128, delta_scale=10.0):
        super().__init__()
        self.gru_dim, self.emb_dim, self.S = gru_dim, emb_dim, float(delta_scale)
        self.cell = nn.GRUCell(8, gru_dim)
        self.vel = mlp(gru_dim + 1, hidden, 4)
        self.emb = nn.Sequential(nn.Linear(gru_dim, emb_dim), nn.LayerNorm(emb_dim))

    def forward(self, hist, hist_len, gap):
        """hist (B,N,K,4), hist_len (B,N), gap (B,N) frames desde la última asignación -> box_pred (B,N,4), vel (B,N,4), m (B,N,emb_dim)."""
        B, N, K, _ = hist.shape
        valid = torch.arange(K, device=hist.device)[None, None, :] >= (K - hist_len[..., None])          # (B,N,K)
        h = hist.new_zeros(B * N, self.gru_dim)
        for k in range(K):
            box, ok = hist[:, :, k], valid[:, :, k]
            has_prev = ok & valid[:, :, k - 1] if k > 0 else torch.zeros_like(ok)
            delta = torch.where(has_prev[..., None], box - hist[:, :, k - 1], torch.zeros_like(box)) * self.S if k > 0 else torch.zeros_like(box)
            h_new = self.cell(torch.cat([box, delta], -1).flatten(0, 1), h)
            h = torch.where(ok.flatten()[:, None], h_new, h)
        h = h.view(B, N, self.gru_dim)
        g = gap.float()
        vel = self.vel(torch.cat([h, g[..., None] / 10], -1)) * MU_SCALE
        box_pred = apply_delta(hist[:, :, -1], vel * g.clamp(min=1)[..., None])
        return box_pred, vel, self.emb(h)
