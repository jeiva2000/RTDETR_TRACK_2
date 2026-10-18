"""Tokens de detección por frame. El criterio de selección (top-K por score) es el MISMO en entrenamiento, caché e inferencia:
topk_queries de tracking/video/extractor.py. En H1 el token es q_det + caja + score; a_det (RoIAlign / puntos deformables) llega en H5."""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tracking.video.extractor import topk_queries  # noqa: F401  (se reexporta: criterio único de selección)


class DetTokens:
    """Tokens de detección de un lote de frames. emb (B,K,D) · box (B,K,4) cxcywh normalizada · score (B,K) · valid (B,K) bool."""

    def __init__(self, emb, box, score, valid=None):
        self.emb, self.box, self.score = emb, box, score
        self.valid = torch.ones_like(score, dtype=torch.bool) if valid is None else valid

    @property
    def shape(self):
        return self.emb.shape[:2]

    def drop(self, mask):
        """Devuelve tokens con las detecciones de `mask` (B,K) marcadas como no válidas (oclusión simulada)."""
        return DetTokens(self.emb, self.box, self.score, self.valid & ~mask)


def from_cache(emb, box, score):
    return DetTokens(emb.float(), box.float(), score.float())
