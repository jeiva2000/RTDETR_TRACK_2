"""Pérdidas y métricas de TrackHeads. Todo se enmascara con track_mask (relleno) y la caja y la atención solo cuentan en tracks presentes.
- box: w_l1·L1 + w_giou·(1 − GIoU) sobre (cx, cy, w, h).
- presence: BCE con logits (pos_weight opcional: en DanceTrack casi todos los tracks siguen presentes).
- attn (auxiliar): −log(masa de la atención dentro de la caja GT de t+k). Los tracks sin ningún token dentro de su caja no cuentan.
  Motivo: con la caja como única señal la atención aprende muy despacio (ver docs/TRACKHEADS.md).
- assoc (la principal desde trackheads_v3): entropía cruzada sobre las detecciones candidatas de t+k + "ninguna" (out['assoc_logits'],
  objetivo tg['assoc_target'], targets.assoc_targets), en todos los tracks (también los ausentes: su objetivo es "ninguna")."""
import torch
import torch.nn.functional as F

from tracking.e2e.h3.geometry import giou


def box_iou(a, b):
    """IoU elemento a elemento, cxcywh (...,4) -> (...)."""
    ax0, ay0, ax1, ay1 = a[..., 0] - a[..., 2] / 2, a[..., 1] - a[..., 3] / 2, a[..., 0] + a[..., 2] / 2, a[..., 1] + a[..., 3] / 2
    bx0, by0, bx1, by1 = b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2, b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2
    inter = (torch.min(ax1, bx1) - torch.max(ax0, bx0)).clamp(min=0) * (torch.min(ay1, by1) - torch.max(ay0, by0)).clamp(min=0)
    return inter / (a[..., 2] * a[..., 3] + b[..., 2] * b[..., 3] - inter + 1e-9)


def attn_mass_in_box(attn, token_xy, box):
    """attn (B,T,N), token_xy (N,2), box (B,T,4) -> masa dentro de la caja (B,T) y nº de tokens dentro (B,T)."""
    x, y = token_xy[:, 0], token_xy[:, 1]
    cx, cy, w, h = (box[..., i, None] for i in range(4))
    inside = ((x - cx).abs() <= w / 2) & ((y - cy).abs() <= h / 2)                    # (B,T,N)
    return (attn * inside).sum(-1), inside.sum(-1)


def trackheads_loss(out, tg, w_l1=5.0, w_giou=2.0, w_pres=1.0, w_attn=1.0, pos_weight=1.0, w_assoc=0.0):
    """out: salida de TrackHeads; tg: build_tracks. -> (pérdida total, dict de partes como floats)."""
    m = tg['track_mask']
    pm = m & (tg['present'] > 0.5)
    zero = out['boxes'].sum() * 0
    if pm.any():
        pb, gb = out['boxes'][pm], tg['target_box'][pm]
        l1 = F.l1_loss(pb, gb, reduction='none').sum(-1).mean()
        lg = (1 - giou(pb, gb)).mean()
    else:
        l1 = lg = zero
    if m.any():
        lp = F.binary_cross_entropy_with_logits(out['presence_logits'][m], tg['present'][m],
                                                pos_weight=out['presence_logits'].new_tensor(pos_weight))
    else:
        lp = zero
    la = zero
    if w_attn > 0 and pm.any():
        mass, n_in = attn_mass_in_box(out['attn'], out['token_xy'], tg['target_box'])
        ok = pm & (n_in > 0)
        if ok.any():
            la = -(mass[ok] + 1e-6).log().mean()
    ls = zero
    if w_assoc > 0 and m.any() and 'assoc_logits' in out:
        ls = F.cross_entropy(out['assoc_logits'][m].float(), tg['assoc_target'][m])
    total = w_l1 * l1 + w_giou * lg + w_pres * lp + w_attn * la + w_assoc * ls
    parts = dict(loss=total, l1=l1, giou=lg, pres=lp, attn=la, assoc=ls)
    return total, {k: v.detach().item() for k, v in parts.items()}


@torch.no_grad()
def batch_stats(out, tg):
    """Sumas (para agregar en la época): IoU de la predicción y de quedarse en prev_box, aciertos de presencia y masa de la atención en la caja."""
    m = tg['track_mask']
    pm = m & (tg['present'] > 0.5)
    iou_p = box_iou(out['boxes'], tg['target_box'])[pm]
    iou_0 = box_iou(tg['prev_box'], tg['target_box'])[pm]
    prob = out['presence_logits'].sigmoid()[m]
    lab = tg['present'][m]
    mass, _ = attn_mass_in_box(out['attn'], out['token_xy'], tg['target_box'])
    return dict(n_tracks=int(m.sum()), n_present=int(pm.sum()), n_absent=int((m & (tg['present'] < 0.5)).sum()),
                iou=float(iou_p.sum()), iou_prev=float(iou_0.sum()), iou50=int((iou_p >= 0.5).sum()), iou50_prev=int((iou_0 >= 0.5).sum()),
                iou75=int((iou_p >= 0.75).sum()), iou75_prev=int((iou_0 >= 0.75).sum()),
                pres_ok=int(((prob >= 0.5).float() == lab).sum()), pres_ok_neg=int(((prob < 0.5) & (lab < 0.5)).sum()),
                attn_mass=float(mass[pm].sum()))


def summarize(acc):
    """Sumas de batch_stats -> métricas (medias)."""
    P, A, N = max(acc['n_present'], 1), max(acc['n_absent'], 1), max(acc['n_tracks'], 1)
    return dict(n_tracks=acc['n_tracks'], present_frac=acc['n_present'] / N,
                iou=acc['iou'] / P, iou_prev=acc['iou_prev'] / P, iou_gain=(acc['iou'] - acc['iou_prev']) / P,
                iou50=acc['iou50'] / P, iou50_prev=acc['iou50_prev'] / P, iou75=acc['iou75'] / P, iou75_prev=acc['iou75_prev'] / P,
                pres_acc=acc['pres_ok'] / N, pres_recall_neg=acc['pres_ok_neg'] / A if acc['n_absent'] else None,
                attn_mass=acc['attn_mass'] / P)
