"""Pérdidas del H3 (sumas y recuentos; la normalización se hace sobre todo el clip en el bucle de entrenamiento).
- L_assign (assignment.sinkhorn_nll): NLL sobre la matriz de Sinkhorn contra la asignación GT, dustbin incluido.
- L_motion: L1 + (1 - GIoU) entre la caja predicha y la caja GT del track en t.
- L_app: InfoNCE sobre a_det y a_trk; positivos = misma ID GT (cada detección con ID presente en memoria, contra todos los tracks)."""
import torch
import torch.nn.functional as F

from tracking.e2e.h3.geometry import giou
from tracking.e2e.h3.relational_attention import NEG


def motion_loss(box_pred, gt_box, mask):
    """box_pred, gt_box (B,N,4), mask (B,N) -> (suma, nº de tracks). Sin tracks: cero conectado al grafo (DDP)."""
    n = mask.sum()
    if n == 0:
        return box_pred.sum() * 0.0, n
    gt = torch.where(mask[..., None], gt_box.detach(), box_pred.detach())               # los huecos sin objetivo no aportan valores extremos
    per = (box_pred - gt).abs().sum(-1) + (1 - giou(box_pred, gt))
    return (per * mask).sum(), n


def app_infonce(a_det, a_trk, tid, ids, det_valid, trk_valid, tau):
    """a_det (B,D,C), a_trk (B,T,C) normalizados; tid (B,D) ID GT de cada detección (-1 = ninguna), ids (B,T) ID de cada track -> (suma, nº de detecciones)."""
    eq = (tid[:, :, None] == ids[:, None, :]) & det_valid[:, :, None] & trk_valid[:, None, :] & (tid[:, :, None] >= 0)
    pos = eq.any(2)
    n = pos.sum()
    if n == 0:
        return (a_det.sum() + a_trk.sum()) * 0.0, n
    logits = (a_det @ a_trk.transpose(1, 2) / tau).masked_fill(~trk_valid[:, None, :], NEG)
    nll = torch.logsumexp(logits, -1) - logits.gather(2, eq.float().argmax(2, keepdim=True)).squeeze(-1)
    return (nll * pos).sum(), n
