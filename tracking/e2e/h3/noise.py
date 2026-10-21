"""Ruido de entrenamiento del H3 (fase 1, detector congelado): las cajas observadas se perturban para parecerse a una detección real y el módulo aprende a ser robusto.
- jitter de posición (fracción del tamaño de la caja) y de escala (log-normal) en todas las detecciones;
- drop: se oculta una detección de un ID GT (oclusión / falso negativo);
- falsos positivos sintéticos: algunas queries de fondo (sin ID GT) pasan a ser detecciones válidas con score 1 y una caja aleatoria.
Devuelve las detecciones con ruido y las cajas LIMPIAS, que son el objetivo de la pérdida de movimiento (la memoria sí guarda las observadas, con ruido)."""
import torch

from tracking.e2e.tokens import DetTokens


def noisy_dets(dets, tid_t, nc, p_drop):
    """dets: DetTokens (B,K), tid_t (B,K) ID GT por query (-1 = fondo) -> (DetTokens con ruido, cajas limpias (B,K,4))."""
    clean = dets.box
    box, score, valid = dets.box.clone(), dets.score.clone(), dets.valid.clone()
    dev, shape = box.device, dets.score.shape
    wh = box[..., 2:4]
    box[..., :2] = box[..., :2] + torch.randn_like(box[..., :2]) * nc.jitter_pos * wh
    box[..., 2:4] = wh * torch.exp(torch.randn_like(wh) * nc.jitter_scale)
    if p_drop > 0:
        valid = valid & ~((tid_t >= 0) & (torch.rand(shape, device=dev) < p_drop))
    if nc.p_fp > 0:
        fp = (tid_t < 0) & (torch.rand(shape, device=dev) < nc.p_fp)
        src = torch.randint(0, shape[1], shape, device=dev)                                                        # el tamaño del FP se copia de otra query
        size = torch.gather(clean[..., 2:4], 1, src[..., None].expand(-1, -1, 2))
        rnd = torch.cat([torch.rand(*shape, 2, device=dev) * 0.9 + 0.05, size], -1)
        box = torch.where(fp[..., None], rnd, box)
        score = torch.where(fp, torch.ones_like(score), score)
        valid = valid | fp
    return DetTokens(dets.emb, box, score, valid), clean
