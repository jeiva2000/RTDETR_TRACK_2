"""Pasos 3, 4 y 5: atención con relaciones espaciales entre objetos.
- Auto-atención de tracks (paso 3) y de detecciones (paso 4): multi-head con SESGO ADITIVO en los logits, B[h,i,j] = MLP_geo(g_ij), con
  g_ij = [Δcx/w_i, Δcy/h_i, log(w_j/w_i), log(h_j/h_i), (Δvx, Δvy)]: i = quien atiende, j = a quien atiende. Los tracks usan su caja PREDICHA (la posición en
  el frame actual, comparable con la de las detecciones) y la velocidad predicha (6 términos); las detecciones, solo los 4 de posición y tamaño. Cada capa:
  residual + LayerNorm + FFN. Opcionalmente un sesgo extra por par (memoria de aristas, fase 2) entra sumado a B.
- Atención cruzada (paso 5): detecciones como queries, tracks como keys/values; devuelve los logits (B,H,D,T) de la capa, que en la última dan s_rel.
Las llaves de relleno se enmascaran con NEG (no -inf): una fila sin llaves válidas da una salida finita (se descarta fuera) y no NaN."""
import math
import torch
import torch.nn as nn

from tracking.e2e.heads import mlp

NEG = -1e4


def pair_geometry(boxes, vel=None):
    """boxes (B,N,4) cxcywh [, vel (B,N,2+)] -> g (B,N,N,4 ó 6) con g[b,i,j] = relación del objeto j respecto al i."""
    cx, cy, w, h = boxes.unbind(-1)
    wi, hi = w.clamp(min=1e-4)[:, :, None], h.clamp(min=1e-4)[:, :, None]
    g = [(cx[:, None, :] - cx[:, :, None]) / wi, (cy[:, None, :] - cy[:, :, None]) / hi,
         torch.log(w.clamp(min=1e-4)[:, None, :] / wi), torch.log(h.clamp(min=1e-4)[:, None, :] / hi)]
    if vel is not None:
        g += [vel[:, None, :, 0] - vel[:, :, None, 0], vel[:, None, :, 1] - vel[:, :, None, 1]]
    return torch.stack(g, -1)


class BiasedAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.h, self.dh = heads, dim // heads
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))

    def forward(self, xq, xkv, key_valid, bias=None):
        """xq (B,Nq,dim), xkv (B,Nk,dim), key_valid (B,Nk) bool, bias (B,H,Nq,Nk) opcional -> (salida (B,Nq,dim), logits (B,H,Nq,Nk))."""
        B, Nq, Nk = xq.shape[0], xq.shape[1], xkv.shape[1]
        sp = lambda x, n: x.view(B, n, self.h, self.dh).transpose(1, 2)
        q, k, v = sp(self.q(xq), Nq), sp(self.k(xkv), Nk), sp(self.v(xkv), Nk)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        if bias is not None:
            logits = logits + bias
        logits = logits.masked_fill(~key_valid[:, None, None, :], NEG)
        out = (torch.softmax(logits, -1) @ v).transpose(1, 2).reshape(B, Nq, self.h * self.dh)
        return self.o(out), logits


class GeoSelfLayer(nn.Module):
    """Auto-atención con sesgo geométrico + residual + LayerNorm + FFN (pasos 3 y 4; pesos propios en cada uno)."""

    def __init__(self, dim, heads, ffn, geo_in, geo_hidden):
        super().__init__()
        self.attn, self.geo = BiasedAttention(dim, heads), mlp(geo_in, geo_hidden, heads)
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.ReLU(inplace=True), nn.Linear(ffn, dim))

    def forward(self, x, valid, geo, extra_bias=None):
        """x (B,N,dim), valid (B,N), geo (B,N,N,geo_in), extra_bias (B,H,N,N) opcional."""
        bias = self.geo(geo).permute(0, 3, 1, 2)
        if extra_bias is not None:
            bias = bias + extra_bias
        x = self.n1(x + self.attn(x, x, valid, bias)[0])
        return self.n2(x + self.ffn(x))


class CrossLayer(nn.Module):
    """Atención cruzada detecciones (queries) <- tracks (keys/values) + residual + LayerNorm + FFN."""

    def __init__(self, dim, heads, ffn):
        super().__init__()
        self.attn = BiasedAttention(dim, heads)
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.ReLU(inplace=True), nn.Linear(ffn, dim))

    def forward(self, xd, xt, trk_valid):
        """xd (B,D,dim), xt (B,T,dim), trk_valid (B,T) -> (xd (B,D,dim), logits (B,H,D,T))."""
        a, logits = self.attn(xd, xt, trk_valid)
        x = self.n1(xd + a)
        return self.n2(x + self.ffn(x)), logits
