"""Tracks y objetivos de un lote de pares (t, t+k).
Tracks: las detecciones del frame t emparejadas con una identidad GT (Hungarian del detector con IoU mínimo) y con score >= det_thr; como mucho
max_tracks por imagen (las de mayor score). emb y prev_box son los del detector (como en inferencia, no la caja GT).
Objetivos: present = la identidad está en el GT de t+k; target_box = su caja GT en t+k (sin significado si no está presente)."""
import torch


def build_tracks(q, tid, gt_tk, det_thr=0.3, max_tracks=64):
    """q: dict con emb (B,K,D), boxes (B,K,4), scores (B,K); tid (B,K) (-1 = fondo); gt_tk lista (B) de (n,5) [id, cx, cy, w, h].
    -> dict(emb (B,T,D), prev_box (B,T,4), track_mask (B,T), present (B,T) float, target_box (B,T,4), track_id (B,T)), T = máx. tracks del lote (>= 1)."""
    emb, boxes, scores = q['emb'], q['boxes'], q['scores']
    B, K, D = emb.shape
    dev = emb.device
    keep = (tid >= 0) & (scores >= det_thr)
    sel = []
    for b in range(B):
        idx = keep[b].nonzero().squeeze(1)
        idx = idx[scores[b, idx].argsort(descending=True)][:max_tracks]
        sel.append(idx)
    T = max(1, max(len(s) for s in sel))
    out = dict(emb=emb.new_zeros(B, T, D), prev_box=boxes.new_tensor([0.5, 0.5, 0.1, 0.1]).repeat(B, T, 1),
               track_mask=torch.zeros(B, T, dtype=torch.bool, device=dev), present=emb.new_zeros(B, T),
               target_box=boxes.new_tensor([0.5, 0.5, 0.1, 0.1]).repeat(B, T, 1), track_id=torch.full((B, T), -1, dtype=torch.long, device=dev))
    for b, idx in enumerate(sel):
        n = len(idx)
        if not n:
            continue
        out['emb'][b, :n], out['prev_box'][b, :n], out['track_mask'][b, :n] = emb[b, idx], boxes[b, idx], True
        ids = tid[b, idx].long()
        out['track_id'][b, :n] = ids
        g = gt_tk[b].to(dev)
        if len(g):
            eq = ids[:, None] == g[:, 0].long()[None]                               # (n, m)
            has = eq.any(1)
            out['present'][b, :n] = has.float()
            out['target_box'][b, :n][has] = g[eq.float().argmax(1)[has], 1:5].float()
    return out


def assoc_targets(track_id, tid_k, cand):
    """track_id (B,T), tid_k (B,C) identidades de las detecciones de t+k (-1 = fondo), cand (B,C) bool (candidatas).
    -> (B,T) long: índice de la candidata con la misma identidad, o C ("ninguna") si no la hay (ausente o no detectada con score suficiente).
    El Hungarian del detector da como mucho una detección por identidad."""
    C = tid_k.shape[1]
    eq = (track_id[:, :, None] == tid_k[:, None, :]) & (tid_k[:, None, :] >= 0) & cand[:, None, :]
    return torch.where(eq.any(-1), eq.float().argmax(-1), torch.full_like(track_id, C))
