"""Predictor de movimiento del H2: segundo vector por track (m, estado oculto de un GRU) junto a la apariencia a.
- Parametrización: un movimiento es el delta de caja respecto a la ÚLTIMA caja observada del track, normalizado por su tamaño:
  (dx/w, dy/h, log w'/w, log h'/h). Es la misma que usa la rama geométrica de heads.geometry_b, así que el delta observado de una detección
  y la predicción son directamente comparables.
- MotionHead.observe: el GRU se actualiza solo con match, con [velocidad (delta/dt), delta, dt/10]; sin match m no cambia y el hueco (gap) crece.
- MotionHead.predict: (m, gap) -> media mu (B,N,4) y log σ (B,N,4) del delta para el frame actual (el gap entra como entrada, así que sirve con saltos).
- motion_features: z = (observado - mu)/σ, NLL gaussiana e IoU con la caja predicha, para cada par track×detección; entran en la afinidad.
- motion_loss: NLL gaussiana del delta de la detección con el MISMO ID (solo tracks presentes), con el estado previo a la actualización."""
import torch
import torch.nn as nn

from tracking.e2e.heads import box_iou_b, mlp

LOGSIG_MIN, LOGSIG_MAX = -4.0, 1.0       # σ entre 0.018 y 2.7 (en unidades de tamaño de caja)
MU_SCALE = 0.1                            # salida cruda O(1) -> delta O(0.1)
EXTRA_DIM = 6                             # z (4) + NLL (1) + IoU con la caja predicha (1)


def box_delta(tbox, obox):
    """Delta de obox respecto a tbox (cxcywh, mismas formas broadcast) -> (...,4)."""
    w, h = tbox[..., 2].clamp(min=1e-4), tbox[..., 3].clamp(min=1e-4)
    return torch.stack([(obox[..., 0] - tbox[..., 0]) / w, (obox[..., 1] - tbox[..., 1]) / h,
                        torch.log(obox[..., 2].clamp(min=1e-4) / w), torch.log(obox[..., 3].clamp(min=1e-4) / h)], -1)


def apply_delta(tbox, d):
    """Inversa de box_delta: caja predicha a partir de tbox y un delta."""
    w, h = tbox[..., 2].clamp(min=1e-4), tbox[..., 3].clamp(min=1e-4)
    return torch.stack([tbox[..., 0] + d[..., 0] * w, tbox[..., 1] + d[..., 1] * h,
                        w * torch.exp(d[..., 2].clamp(-3, 3)), h * torch.exp(d[..., 3].clamp(-3, 3))], -1)


class MotionHead(nn.Module):
    def __init__(self, dim, hidden, vel_scale=10.0):
        super().__init__()
        self.dim, self.S = dim, float(vel_scale)
        self.cell = nn.GRUCell(9, dim)
        self.pred = mlp(dim + 1, hidden, 8)
        with torch.no_grad():
            self.pred[-1].bias[4:] = -1.5     # σ inicial ~0.22

    def observe(self, m, tbox, obox, dt, matched):
        """m (B,N,M), tbox (B,N,4) caja previa del track, obox (B,N,4) caja observada, dt (B,N) frames desde la última vez, matched (B,N)."""
        delta = torch.where(matched[..., None], box_delta(tbox, obox), torch.zeros_like(tbox))   # relleno/sin match: entrada 0 (no se usa)
        dtf = dt.float().clamp(min=1)[..., None]
        x = torch.cat([delta / dtf * self.S, delta * self.S, dtf / 10], -1)
        m_new = self.cell(x.flatten(0, 1), m.flatten(0, 1)).view_as(m)
        return torch.where(matched[..., None], m_new, m)

    def predict(self, m, gap):
        out = self.pred(torch.cat([m, gap.float()[..., None] / 10], -1))
        return out[..., :4] * MU_SCALE, out[..., 4:].clamp(LOGSIG_MIN, LOGSIG_MAX)


def motion_features(tbox, mu, logsig, dbox):
    """tbox (B,N,4) mu/logsig (B,N,4) dbox (B,K,4) -> (B,N,K,EXTRA_DIM)."""
    obs = box_delta(tbox[:, :, None], dbox[:, None])
    ls = logsig[:, :, None]
    z = ((obs - mu[:, :, None]) / ls.exp()).clamp(-10, 10)
    nll = (0.5 * z ** 2 + ls).sum(-1, keepdim=True) / 4
    iou = box_iou_b(apply_delta(tbox, mu), dbox)[..., None]
    return torch.cat([z / 3, nll, iou], -1)


def motion_loss(mu, logsig, obs, mask):
    """NLL gaussiana (suma, nº de tracks). mu/logsig/obs (B,N,4), mask (B,N). Sin tracks: cero CONECTADO al grafo (DDP)."""
    n = mask.sum()
    if n == 0:
        return (mu.sum() + logsig.sum()) * 0.0, n
    obs = torch.where(mask[..., None], obs.detach(), mu.detach())          # los huecos sin objetivo no aportan valores extremos
    nll = (0.5 * ((obs - mu) / logsig.exp()) ** 2 + logsig).sum(-1)
    return (nll * mask).sum(), n
