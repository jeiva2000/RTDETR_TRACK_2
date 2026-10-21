"""Paso 6: afinidad con gates por par. Tres señales por par detección i – track j:
  s_app = coseno(a_det, a_trk) · s_mot = GIoU(caja detección, caja predicha) · s_rel = logits de la atención cruzada (media entre heads).
Entrada de los gates de cada par: [s_app, s_mot, s_rel, margen_app_i, margen_mot_i, margen_rel_i], con margen_x_i = top1 - top2 de la fila i de esa señal
sobre los tracks válidos (mide la ambigüedad de la detección; 0 si hay < 2 tracks). MLP(6 -> 32 -> 3) + softmax -> (g_a, g_m, g_r);
s_ij = temperatura · (g_a·s_app + g_m·s_mot + g_r·s_rel), con temperatura aprendible.
Ablaciones: `branches` (app, mot, rel) desactiva ramas (su señal vale 0 y el softmax solo reparte entre las activas) y use_gates=False usa pesos fijos."""
import torch
import torch.nn as nn

from tracking.e2e.h3.relational_attention import NEG
from tracking.e2e.heads import mlp


def row_margin(s, trk_valid):
    """top1 - top2 de cada fila (B,D,T) sobre los tracks válidos -> (B,D); 0 con menos de 2 tracks válidos."""
    B, D, T = s.shape
    if T < 2:
        return s.new_zeros(B, D)
    top = s.masked_fill(~trk_valid[:, None, :], NEG).topk(2, dim=-1).values
    return torch.where(top[..., 1] > NEG / 2, top[..., 0] - top[..., 1], torch.zeros_like(top[..., 0]))


class AffinityGates(nn.Module):
    def __init__(self, branches=(True, True, True), use_gates=True, fixed_w=(1 / 3, 1 / 3, 1 / 3), tau_init=2.3, hidden=32):
        super().__init__()
        assert any(branches), 'al menos una rama (app, mot, rel) debe estar activa'
        self.use_gates = use_gates
        self.register_buffer('active', torch.tensor(list(branches), dtype=torch.bool))
        self.register_buffer('fixed_w', torch.tensor(list(fixed_w), dtype=torch.float32))
        self.gate = mlp(6, hidden, 3)
        self.log_tau = nn.Parameter(torch.tensor(float(tau_init)))      # exp(2.3) ~ 10

    def forward(self, s_app, s_mot, s_rel, trk_valid):
        """s_* (B,D,T), trk_valid (B,T) -> (s (B,D,T) con temperatura, gates (B,D,T,3))."""
        B, D, T = s_app.shape
        sig = torch.stack([s_app, s_mot, s_rel], -1) * self.active                                           # ramas inactivas: 0
        if self.use_gates:
            marg = torch.stack([row_margin(sig[..., k], trk_valid) for k in range(3)], -1)                  # (B,D,3)
            x = torch.cat([sig, marg[:, :, None, :].expand(B, D, T, 3)], -1)
            g = torch.softmax(self.gate(x).masked_fill(~self.active, NEG), -1)
        else:
            w = self.fixed_w * self.active
            g = (w / w.sum()).expand(B, D, T, 3)
        return self.log_tau.exp() * (g * sig).sum(-1), g
