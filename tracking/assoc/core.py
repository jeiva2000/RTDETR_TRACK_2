"""Paso de asociación común a entrenamiento e inferencia (mismo código para ambos)."""
import torch
from .model import box_iou_cxcywh


class Tracks:
    """Tracks activos o perdidos (<= max_lost frames). ids: identificador (GT en entrenamiento, propio en inferencia).
    Con hist > 0 cada track guarda además un historial FIFO de sus últimas observaciones (la más reciente en la última
    posición): proyección p, caja, frame, confianza y máscara de validez; y la última visibilidad predicha (logits)."""

    def __init__(self, dim, device, hist=0):
        self.ids, self.dim, self.device, self.H = [], dim, device, hist
        self.S = torch.zeros(0, dim, device=device)
        self.tb = torch.zeros(0, 4, device=device)
        self.last = torch.zeros(0, dtype=torch.long, device=device)
        self.vis = torch.zeros(0, 3, device=device)
        if hist:
            self.Hp = torch.zeros(0, hist, dim, device=device); self.Hb = torch.zeros(0, hist, 4, device=device)
            self.Hf = torch.zeros(0, hist, dtype=torch.long, device=device); self.Hc = torch.zeros(0, hist, device=device)
            self.Hm = torch.zeros(0, hist, dtype=torch.bool, device=device)

    def __len__(self): return len(self.ids)

    def add(self, ids, S, boxes, frame, p=None):
        n = len(ids)
        self.ids += list(ids)
        self.S = torch.cat([self.S, S]); self.tb = torch.cat([self.tb, boxes])
        self.last = torch.cat([self.last, torch.full((n,), int(frame), dtype=torch.long, device=self.device)])
        self.vis = torch.cat([self.vis, torch.zeros(n, 3, device=self.device)])
        if self.H:  # la primera observación entra en el historial en el nacimiento
            assert p is not None, 'Tracks con historial necesita p al nacer'
            z = lambda *sh, **kw: torch.zeros(n, *sh, device=self.device, **kw)
            Hp, Hb, Hc = z(self.H, self.dim), z(self.H, 4), z(self.H)
            Hf, Hm = z(self.H, dtype=torch.long), z(self.H, dtype=torch.bool)
            Hp[:, -1], Hb[:, -1], Hf[:, -1], Hc[:, -1], Hm[:, -1] = p, boxes, int(frame), 1.0, True
            self.Hp, self.Hb, self.Hf = torch.cat([self.Hp, Hp]), torch.cat([self.Hb, Hb]), torch.cat([self.Hf, Hf])
            self.Hc, self.Hm = torch.cat([self.Hc, Hc]), torch.cat([self.Hm, Hm])

    def keep(self, mask):
        self.ids = [i for i, k in zip(self.ids, mask.tolist()) if k]
        self.S, self.tb, self.last, self.vis = self.S[mask], self.tb[mask], self.last[mask], self.vis[mask]
        if self.H:
            self.Hp, self.Hb, self.Hf, self.Hc, self.Hm = (x[mask] for x in (self.Hp, self.Hb, self.Hf, self.Hc, self.Hm))

    def hist_view(self):
        if not self.H:
            return None
        return dict(Hp=self.Hp, Hb=self.Hb, Hf=self.Hf, Hc=self.Hc, Hm=self.Hm)

    def write(self, wmask, p_new, box_new, frame, conf):
        """Añade (desplazando la más antigua) la observación actual a los tracks con wmask=True."""
        if not self.H or not bool(wmask.any()):
            return
        sh = lambda old, new: torch.cat([old[:, 1:], new[:, None]], 1)
        w3 = wmask[:, None, None]
        self.Hp = torch.where(w3, sh(self.Hp, p_new), self.Hp)
        self.Hb = torch.where(w3, sh(self.Hb, box_new.detach()), self.Hb)
        self.Hf = torch.where(wmask[:, None], sh(self.Hf, torch.full_like(self.last, int(frame))), self.Hf)
        self.Hc = torch.where(wmask[:, None], sh(self.Hc, conf.detach()), self.Hc)
        self.Hm = torch.where(wmask[:, None], sh(self.Hm, torch.ones_like(wmask)), self.Hm)


def step(model, tr, E, D, sc, frame, choose, valid=None):
    """Afinidad + elección (choose(logits) -> idx, conf) + actualización del estado. Devuelve logits (N,K+1) e idx (N,).
    Con la variante hist deja además en tr.vis los logits de visibilidad (N,3) de este frame."""
    is_hist = model.update_type == 'hist'
    gap = frame - tr.last
    hist = tr.hist_view()
    logits, geo, p = model.affinity(tr.S, tr.tb, gap, E, D, sc, valid, hist=hist)
    idx, conf = choose(logits)
    idx, conf = idx.to(E.device), conf.to(E.device).float()
    obs = idx >= 0
    new_b = torch.where(obs[:, None], D[idx.clamp(min=0)], tr.tb)
    iou = box_iou_cxcywh(new_b, tr.tb)
    iou.fill_diagonal_(0)
    overlap = (iou.max(1).values if len(tr) > 1 else torch.zeros(len(tr), device=E.device)).detach()
    vis_logits = vis = None
    if is_hist:
        vis_logits = model.visibility(tr.S, gap, overlap, conf.detach(), obs)
        vis = torch.softmax(vis_logits, -1)
    tr.S = model.update(tr.S, idx, conf, overlap, E, geo, p, tb=new_b.detach(), gap=torch.where(obs, torch.zeros_like(gap), gap),
                        vis=vis, hist=hist, frame=frame)
    if is_hist:
        tr.vis = vis_logits
        wmask = obs & (conf.detach() >= model.write_conf) & (overlap <= model.write_overlap)  # regla fija de escritura
        tr.write(wmask, p[idx.clamp(min=0)], new_b, frame, conf)
    tr.tb = new_b
    tr.last = torch.where(obs, torch.full_like(tr.last, int(frame)), tr.last)
    return logits, idx
