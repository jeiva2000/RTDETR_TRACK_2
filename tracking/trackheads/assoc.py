"""Acierto de asociación de TrackHeads: lo que importa en seguimiento no es afinar la caja (la del detector en t+k siempre será mejor), sino
que la cabeza apunte a la detección CORRECTA de t+k.
Para cada track presente cuya identidad está detectada en t+k (candidatas = detecciones de t+k con score >= det_thr, también los falsos
positivos), se elige una candidata con tres reglas y se mira si es la misma identidad:
- pred: la de mayor IoU con la caja predicha por la cabeza.
- prev: la de mayor IoU con prev_box (la caja del frame t): un tracker por IoU sin movimiento, la referencia a batir.
- attn: la que contiene más masa de la atención de la cabeza (usa la atención directamente como señal de asociación).
cruce: la caja GT del track solapa (IoU > cross_iou) con la de otra identidad en t o en t+k; es donde prev falla y la apariencia debería ayudar."""
import torch

from tracking.trackheads.losses import box_iou


def _cross(ids, gt, thr):
    """ids (T,), gt (n,5) -> (T,) bool: la caja GT de esa identidad solapa con otra identidad por encima de thr."""
    if len(gt) < 2:
        return torch.zeros_like(ids, dtype=torch.bool)
    b = gt[:, 1:5].float()
    m = box_iou(b[:, None], b[None])
    m.fill_diagonal_(0)
    eq = ids[:, None] == gt[:, 0].long()[None]
    return (eq.float() * m.amax(1)[None]).amax(1) > thr


def gt_cross(ids, gt_t, gt_tk, thr=0.3):
    """ids (B,T) identidades de los tracks; gt_t, gt_tk listas (B) de (n,5) -> (B,T) bool: en cruce en t o en t+k."""
    return torch.stack([_cross(ids[b], gt_t[b].to(ids.device), thr) | _cross(ids[b], gt_tk[b].to(ids.device), thr) for b in range(ids.shape[0])])


@torch.no_grad()
def assoc_flags(out, tg, box_k, sc_k, tid_k, gt_t, gt_tk, det_thr=0.3, cross_iou=0.3):
    """out: salida de TrackHeads; tg: build_tracks; box_k (B,C,4), sc_k (B,C), tid_k (B,C) detecciones de t+k; gt_t, gt_tk listas (B) de (n,5).
    -> dict de (B,T) bool: valid (track presente con su identidad detectada en t+k), ok_pred, ok_prev, ok_attn, cross."""
    cand = sc_k >= det_thr                                                           # (B,C)
    ids = tg['track_id']
    is_true = (ids[:, :, None] == tid_k[:, None, :]) & (tid_k[:, None, :] >= 0) & cand[:, None, :]   # (B,T,C)
    valid = tg['track_mask'] & (tg['present'] > 0.5) & is_true.any(-1)

    def pick(score):
        j = score.masked_fill(~cand[:, None, :], float('-inf')).argmax(-1)
        return is_true.gather(-1, j[..., None]).squeeze(-1) & valid

    x, y = out['token_xy'][:, 0], out['token_xy'][:, 1]
    inside = ((x[None, None] - box_k[..., 0, None]).abs() <= box_k[..., 2, None] / 2) & \
             ((y[None, None] - box_k[..., 1, None]).abs() <= box_k[..., 3, None] / 2)          # (B,C,N)
    mass = torch.bmm(out['attn'], inside.transpose(1, 2).to(out['attn'].dtype))               # (B,T,C)
    cross = gt_cross(ids, gt_t, gt_tk, cross_iou)
    return dict(valid=valid, ok_pred=pick(box_iou(out['boxes'][:, :, None], box_k[:, None])),
                ok_prev=pick(box_iou(tg['prev_box'][:, :, None], box_k[:, None])), ok_attn=pick(mass), cross=cross & valid)


def assoc_sums(f):
    """Sumas para agregar en la época (total y cruces)."""
    v, c = f['valid'], f['cross']
    s = dict(assoc_n=int(v.sum()), cross_n=int(c.sum()))
    for r in ('pred', 'prev', 'attn'):
        s[f'assoc_{r}'] = int(f[f'ok_{r}'].sum())
        s[f'cross_{r}'] = int((f[f'ok_{r}'] & c).sum())
    return s


def assoc_summary(acc):
    """Sumas -> aciertos (fracciones). None si no hay casos."""
    out = dict(assoc_n=acc.get('assoc_n', 0), cross_n=acc.get('cross_n', 0))
    for g in ('assoc', 'cross'):
        n = acc.get(f'{g}_n', 0)
        for r in ('pred', 'prev', 'attn'):
            out[f'{g}_{r}'] = acc[f'{g}_{r}'] / n if n else None
        out[f'{g}_gain'] = (acc[f'{g}_pred'] - acc[f'{g}_prev']) / n if n else None
    return out
