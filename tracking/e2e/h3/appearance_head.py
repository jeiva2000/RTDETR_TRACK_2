"""Paso 1: proyección de apariencia. MLP(256 -> 256 -> 128) + normalización L2; se aplica igual a los embeddings de las detecciones y a la
apariencia guardada de los tracks (memoria en el espacio del decoder, 256)."""
import torch.nn as nn
import torch.nn.functional as F


class AppearanceHead(nn.Module):
    def __init__(self, in_dim=256, hidden=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out_dim))

    def forward(self, x):
        """x (..., in_dim) -> (..., out_dim) con norma L2 = 1 (los huecos de relleno, con x = 0, dan un vector finito)."""
        return F.normalize(self.net(x), dim=-1)
