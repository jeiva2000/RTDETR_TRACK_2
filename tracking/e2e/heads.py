"""Cabezas del H1, todas vectorizadas por lote (B clips, N tracks, K detecciones) con máscaras:
- AffinityHead: afinidad track×detección = temperatura·coseno(apariencia) + rama geométrica; columna nula ("no aparece") por track.
  Softmax por filas (cada track reparte su probabilidad entre las K detecciones y su columna nula).
- BirthHead: probabilidad de que una detección corresponda a un ID nuevo (BCE aparte; no se mezcla con la softmax de los tracks)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

GEO_DIM = 7


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(inplace=True), nn.Linear(h, o))


def box_iou_b(a, b):
    """a (B,N,4) b (B,K,4) en cxcywh -> IoU (B,N,K)"""
    ax1, ay1, ax2, ay2 = a[..., 0] - a[..., 2] / 2, a[..., 1] - a[..., 3] / 2, a[..., 0] + a[..., 2] / 2, a[..., 1] + a[..., 3] / 2
    bx1, by1, bx2, by2 = b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2, b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2
    iw = (torch.min(ax2[:, :, None], bx2[:, None, :]) - torch.max(ax1[:, :, None], bx1[:, None, :])).clamp(min=0)
    ih = (torch.min(ay2[:, :, None], by2[:, None, :]) - torch.max(ay1[:, :, None], by1[:, None, :])).clamp(min=0)
    inter = iw * ih
    return inter / ((a[..., 2] * a[..., 3])[:, :, None] + (b[..., 2] * b[..., 3])[:, None, :] - inter + 1e-9)


def geometry_b(tbox, gap, dbox, dscore):
    """Rasgos geométricos track->detección (B,N,K,7): dx/w, dy/h, log(w'/w), log(h'/h), IoU, score, frames sin ver/10."""
    w, h = tbox[..., 2].clamp(min=1e-4)[:, :, None], tbox[..., 3].clamp(min=1e-4)[:, :, None]
    dx = (dbox[:, None, :, 0] - tbox[:, :, None, 0]) / w
    dy = (dbox[:, None, :, 1] - tbox[:, :, None, 1]) / h
    lw = torch.log(dbox[:, None, :, 2].clamp(min=1e-4) / w)
    lh = torch.log(dbox[:, None, :, 3].clamp(min=1e-4) / h)
    iou = box_iou_b(tbox, dbox)
    B, N, K = iou.shape
    return torch.stack([dx, dy, lw, lh, iou, dscore[:, None, :].expand(B, N, K), (gap.float() / 10)[:, :, None].expand(B, N, K)], -1)


class AffinityHead(nn.Module):
    def __init__(self, dim, hidden, tau_init=2.3):
        super().__init__()
        self.q_s, self.q_p = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.log_tau = nn.Parameter(torch.tensor(float(tau_init)))      # exp(2.3) ~ 10
        self.geo = mlp(GEO_DIM, 64, 1)
        self.null = mlp(dim + 1, hidden, 1)                             # columna nula: [a, frames sin ver/10]

    def forward(self, a, tbox, gap, p, dbox, dscore, dvalid):
        """a (B,N,D) tbox (B,N,4) gap (B,N) | p (B,K,D) dbox (B,K,4) dscore (B,K) dvalid (B,K) -> logits (B,N,K+1)."""
        app = self.log_tau.exp() * F.normalize(self.q_s(a), dim=-1) @ F.normalize(self.q_p(p), dim=-1).transpose(1, 2)
        logit = (app + self.geo(geometry_b(tbox, gap, dbox, dscore)).squeeze(-1)).masked_fill(~dvalid[:, None, :], float('-inf'))
        null = self.null(torch.cat([a, (gap.float() / 10)[..., None]], -1))
        return torch.cat([logit, null], -1)


class BirthHead(nn.Module):
    """Logit de nacimiento por detección a partir de [p, score, mejor IoU con un track vivo, mayor probabilidad con que algún track la reclama]."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.net = mlp(dim + 3, hidden, 1)

    def forward(self, p, score, iou_max, claim):
        return self.net(torch.cat([p, score[..., None], iou_max[..., None], claim[..., None]], -1)).squeeze(-1)
