"""Banco histórico de apariencias del H3 (h3.bank): lo que el track "recuerda" de cómo se ha visto, más allá de un vector medio. La memoria (track_memory.py) guarda, por track, los últimos M
embeddings fiables; aquí se usan de dos maneras:
- HistoryPool: el token del track pregunta a su propio banco (atención multi-head con la apariencia actual como query y una codificación de la EDAD de cada embedding) y obtiene un resumen del
  historial (B,N,C) que se concatena al token ANTES de la auto-atención entre tracks (paso 3): así el contexto relacional de cada track ya incluye cómo se ha visto. Sin embeddings válidos, el resumen es 0.
- bank_affinity: s_app de cada pareja detección–track = máximo (o media) del coseno contra el banco y la EMA, de modo que un track con varias poses/aspectos tiene más de una referencia
  (como la variante hist_aff=max del módulo antiguo tracking/assoc). Durante una oclusión el banco no se escribe: conserva las apariencias de antes de ocluirse.
Todo con relleno y máscaras (huecos libres y bancos con menos de M embeddings); funciona con D = 0 y T = 0."""
import math
import torch
import torch.nn as nn

from tracking.e2e.h3.relational_attention import NEG


def bank_valid(bank_n, M):
    """(B,N) nº de embeddings válidos -> (B,N,M) bool; los válidos son los últimos (el más reciente al final)."""
    return torch.arange(M, device=bank_n.device)[None, None, :] >= (M - bank_n[..., None])


class HistoryPool(nn.Module):
    def __init__(self, dim=128, heads=4, max_age=60.0):
        super().__init__()
        assert dim % heads == 0
        self.h, self.dh, self.max_age = heads, dim // heads, float(max_age)
        self.age = nn.Sequential(nn.Linear(1, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))      # codificación de la edad (frames desde que se guardó)
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))

    def forward(self, a_trk, a_bank, age, valid):
        """a_trk (B,N,C) apariencia actual, a_bank (B,N,M,C), age (B,N,M) frames, valid (B,N,M) -> resumen (B,N,C)."""
        B, N, M, C = a_bank.shape
        kv = a_bank + self.age((age.float().clamp(max=self.max_age) / self.max_age)[..., None])
        q = self.q(a_trk).view(B, N, 1, self.h, self.dh).transpose(2, 3)                            # (B,N,H,1,dh)
        k = self.k(kv).view(B, N, M, self.h, self.dh).transpose(2, 3)                               # (B,N,H,M,dh)
        v = self.v(kv).view(B, N, M, self.h, self.dh).transpose(2, 3)
        logits = (q @ k.transpose(-1, -2) / math.sqrt(self.dh)).masked_fill(~valid[:, :, None, None, :], NEG)
        out = (torch.softmax(logits, -1) @ v).transpose(2, 3).reshape(B, N, C)
        return self.o(out) * valid.any(-1, keepdim=True)                                             # sin embeddings válidos: resumen 0


def bank_affinity(a_det, a_trk, a_bank, valid, mode='max'):
    """a_det (B,D,C), a_trk (B,N,C), a_bank (B,N,M,C), valid (B,N,M) -> (s_app (B,D,N) agregada, s_ema (B,D,N) solo con la EMA). mode: max | mean (sobre banco y EMA) | none (solo EMA)."""
    s_ema = a_det @ a_trk.transpose(1, 2)
    if mode == 'none' or a_bank.shape[2] == 0:
        return s_ema, s_ema
    cos = torch.einsum('bdc,bnmc->bdnm', a_det, a_bank)                                              # (B,D,N,M)
    v = valid[:, None, :, :]
    if mode == 'max':
        return torch.maximum(s_ema, cos.masked_fill(~v, -2.0).max(-1).values), s_ema
    n = v.sum(-1)
    return (s_ema + (cos * v).sum(-1)) / (1 + n), s_ema
