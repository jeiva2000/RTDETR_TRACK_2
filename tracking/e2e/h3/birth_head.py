"""Cabeza de nacimiento del H3: probabilidad de que una detección deba crear un track nuevo. Sustituye a la regla "detección sin asignar con score alto", que ante cualquier rechazo
creaba un duplicado (el bucle de duplicados del diagnóstico). Es un componente APRENDIDO, así que se puede penalizar con una BCE (losses.birth_loss), como la BirthHead del H1/H2.
Entradas por detección (todas desconectadas del grafo por defecto, para no perjudicar las señales de la asociación):
  resumen de la apariencia (a_det proyectada), score, masa del dustbin P_i,dustbin, mayor probabilidad hacia algún track vivo (el "reclamo"), IoU máximo con las cajas predichas de
  los tracks vivos, afinidad s máxima y nº de tracks vivos (sin tracks, p. ej. al empezar un vídeo, todo es nuevo).
Etiqueta: 1 solo si la detección es de una identidad GT SIN track vivo; 0 para falsos positivos y para identidades que ya tienen track (aunque el modelo no las asigne): esos nacimientos son
los duplicados. La columna dustbin sola no sirve: mezcla identidad nueva y falso positivo."""
import torch
import torch.nn as nn

from tracking.e2e.heads import mlp

N_FEATURES = 6        # score, P_dustbin, reclamo, IoU máximo, s máxima, nº de tracks vivos


class H3BirthHead(nn.Module):
    def __init__(self, app_dim=128, hidden=128, proj=32):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(app_dim, proj), nn.ReLU(inplace=True))
        self.net = mlp(proj + N_FEATURES, hidden, 1)

    def forward(self, a_det, score, p_bin, claim, iou_max, s_max, n_alive):
        """a_det (B,D,C); score, p_bin, claim, iou_max, s_max, n_alive (B,D) -> logit (B,D)."""
        x = torch.cat([self.proj(a_det), torch.stack([score, p_bin, claim, iou_max, s_max, n_alive], -1)], -1)
        return self.net(x).squeeze(-1)
