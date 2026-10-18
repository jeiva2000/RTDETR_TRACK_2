"""Pérdidas del H1 (sumas y recuentos; la normalización se hace sobre todo el clip en el bucle de entrenamiento).
- assoc_loss: entropía cruzada sobre la matriz con columna nula (objetivo: la detección con el mismo ID, o la columna nula).
- birth_loss: BCE de la cabeza de nacimiento (positivo = detección de un ID GT sin track)."""
import torch
import torch.nn.functional as F

IGNORE = -100


def assoc_loss(logits, target):
    """logits (B,N,K+1), target (B,N) con IGNORE en huecos sin track. Devuelve (suma, nº de decisiones)."""
    n = (target != IGNORE).sum()
    if n == 0:   # sin decisiones (p. ej. primer frame del clip): cero CONECTADO al grafo (DDP necesita que todos los parámetros reciban gradiente).
        #           logits.sum() * 0 daría NaN con columnas -inf; nan_to_num las sustituye por 0 antes de sumar.
        return torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0, n
    return F.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=IGNORE, reduction='sum'), n


def birth_loss(logit, label, mask, pos_weight=1.0):
    """logit/label/mask (B,K). Devuelve (suma, nº de detecciones válidas)."""
    pw = torch.tensor(float(pos_weight), device=logit.device)
    bce = F.binary_cross_entropy_with_logits(logit, label.float(), pos_weight=pw, reduction='none')
    return (bce * mask).sum(), mask.sum()
