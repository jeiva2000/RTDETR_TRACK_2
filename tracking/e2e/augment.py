"""Aumento de intercambio de IDs para el entrenamiento de H1/H2 (adaptación de los Spatial ID Switches de MOTIP2).

En MOTIP2 se intercambian, en el historial de trayectorias, las etiquetas de ID de objetos cercanos (Hungarian sobre GIoU + ruido
gaussiano, σ = 0.5), y el objetivo del frame actual sigue siendo la identidad verdadera: el modelo aprende a no fiarse ciegamente de un
historial que puede tener errores. Es el aumento que más pesa en su ablación sobre DanceTrack val.

En H1/H2 el historial es el estado recurrente del track (EMA de apariencia, última caja, GRU de movimiento). El equivalente es corromper
la ACTUALIZACIÓN: en un frame, algunos tracks con match se actualizan con la detección de otro track (la apariencia del vecino entra en la
EMA, la caja salta a la del vecino y el GRU observa ese salto), mientras state.ids no cambia, así que en los frames siguientes el objetivo
de la asociación sigue siendo la persona verdadera. Es exactamente el estado que deja un robo de identidad del propio tracker, y el
modelo aprende a volver a su persona en lugar de quedarse con el vecino.

id_switch: cada track vivo con match entra con probabilidad p; entre los elegidos (si hay al menos 2) se reparte una permutación SIN
puntos fijos de sus detecciones, elegida por Hungarian sobre GIoU + N(0, σ²) con la diagonal prohibida: los intercambios ocurren sobre
todo entre vecinos, con algo de aleatoriedad. Devuelve la asignación alterada y la máscara de tracks alterados."""
import torch
from scipy.optimize import linear_sum_assignment


def giou_cxcywh(a, b):
    """a (n,4), b (m,4) en cxcywh -> GIoU (n,m) en [-1, 1]."""
    def xyxy(x):
        return torch.cat([x[:, :2] - x[:, 2:] / 2, x[:, :2] + x[:, 2:] / 2], -1)
    a, b = xyxy(a), xyxy(b)
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt, rb = torch.max(a[:, None, :2], b[None, :, :2]), torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = area_a[:, None] + area_b[None] - inter
    iou = inter / union.clamp(min=1e-9)
    lt_c, rb_c = torch.min(a[:, None, :2], b[None, :, :2]), torch.max(a[:, None, 2:], b[None, :, 2:])
    hull = (rb_c - lt_c).clamp(min=0).prod(-1)
    return iou - (hull - union) / hull.clamp(min=1e-9)


@torch.no_grad()
def id_switch(assign, alive, box, p, sigma=0.5, generator=None):
    """assign (B,N) en [0, K] (K = "no aparece"), alive (B,N), box (B,K,4) cxcywh de las detecciones del frame.
    -> (assign alterada, swapped (B,N) bool). Solo se tocan tracks vivos con match; las detecciones asignadas se permutan entre ellos."""
    B, N = assign.shape
    K = box.shape[1]
    out, swapped = assign.clone(), torch.zeros_like(alive)
    if p <= 0:
        return out, swapped
    matched = alive & (assign < K)
    sel = matched & (torch.rand(B, N, generator=generator).to(assign.device) < p)
    for b in range(B):
        rows = sel[b].nonzero().flatten()
        n = len(rows)
        if n < 2:
            continue
        j = assign[b, rows]
        s = giou_cxcywh(box[b, j].float(), box[b, j].float()).cpu()
        if sigma > 0:
            s = s + sigma * torch.randn(n, n, generator=generator)
        s.fill_diagonal_(-1e6)                                   # nadie se queda con su detección: permutación sin puntos fijos
        r, c = linear_sum_assignment(-s.numpy())
        out[b, rows[torch.as_tensor(r, device=rows.device)]] = j[torch.as_tensor(c, device=j.device)]
        swapped[b, rows] = True
    return out, swapped
