"""Pérdidas del H3 (sumas y recuentos; la normalización se hace sobre todo el clip en el bucle de entrenamiento).
- L_assign (assignment.sinkhorn_nll): NLL sobre la matriz de Sinkhorn contra la asignación GT, dustbin incluido.
- L_motion: L1 + (1 - GIoU) entre la caja predicha y la caja GT del track en t.
- L_app: InfoNCE sobre a_det y a_trk; positivo = el hueco que debe recibir la detección según el objetivo de la asignación (misma identidad GT), contra todos los tracks."""
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


def app_infonce(a_det, a_trk, pos_idx, pos_mask, trk_valid, tau):
    """a_det (B,D,C), a_trk (B,T,C) normalizados; pos_idx (B,D) hueco que debe recibir cada detección (el objetivo de la asignación: con duplicados, el más antiguo de la identidad),
    pos_mask (B,D) detecciones con positivo, trk_valid (B,T) -> (suma, nº de detecciones). Sin positivos: cero conectado al grafo (DDP)."""
    n = pos_mask.sum()
    if n == 0:
        return (a_det.sum() + a_trk.sum()) * 0.0, n
    logits = (a_det @ a_trk.transpose(1, 2) / tau).masked_fill(~trk_valid[:, None, :], NEG)
    nll = torch.logsumexp(logits, -1) - logits.gather(2, pos_idx.clamp(max=a_trk.shape[1] - 1)[..., None]).squeeze(-1)
    return (nll * pos_mask).sum(), n
