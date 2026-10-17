"""Paso de asociación común a entrenamiento e inferencia (mismo código para ambos)."""
import torch
from .model import box_iou_cxcywh


class Tracks:
    """Tracks activos o perdidos (<= max_lost frames). ids: identificador (GT en entrenamiento, propio en inferencia)."""

    def __init__(self, dim, device):
        self.ids, self.dim, self.device = [], dim, device
        self.S = torch.zeros(0, dim, device=device)
        self.tb = torch.zeros(0, 4, device=device)
        self.last = torch.zeros(0, dtype=torch.long, device=device)

    def __len__(self): return len(self.ids)

    def add(self, ids, S, boxes, frame):
        self.ids += list(ids)
        self.S = torch.cat([self.S, S]); self.tb = torch.cat([self.tb, boxes])
        self.last = torch.cat([self.last, torch.full((len(ids),), int(frame), dtype=torch.long, device=self.device)])

    def keep(self, mask):
        self.ids = [i for i, k in zip(self.ids, mask.tolist()) if k]
        self.S, self.tb, self.last = self.S[mask], self.tb[mask], self.last[mask]


def step(model, tr, E, D, sc, frame, choose, valid=None):
    """Afinidad + elección (choose(logits) -> idx, conf) + actualización del estado. Devuelve logits (N,K+1) e idx (N,)."""
    gap = frame - tr.last
    logits, geo, p = model.affinity(tr.S, tr.tb, gap, E, D, sc, valid)
    idx, conf = choose(logits)
    idx, conf = idx.to(E.device), conf.to(E.device).float()
    obs = idx >= 0
    new_b = torch.where(obs[:, None], D[idx.clamp(min=0)], tr.tb)
    iou = box_iou_cxcywh(new_b, tr.tb)
    iou.fill_diagonal_(0)
    overlap = iou.max(1).values if len(tr) > 1 else torch.zeros(len(tr), device=E.device)
    tr.S = model.update(tr.S, idx, conf, overlap.detach(), E, geo, p, tb=new_b.detach(), gap=torch.where(obs, torch.zeros_like(gap), gap))
    tr.tb = new_b
    tr.last = torch.where(obs, torch.full_like(tr.last, int(frame)), tr.last)
    return logits, idx
