"""Paso 8: memoria de tracks en tensores con relleno y máscaras (B clips, N huecos). Cada track guarda: apariencia app (espacio del decoder, 256), historial de las últimas
K cajas (la más reciente en el índice -1; hist_len estados válidos), id, frame de nacimiento (edad), último frame asignado (frames desde la última asignación) y alive.
Operaciones fuera de lugar (seguras con autograd/BPTT). Un track sin asignar sigue en memoria (para permitir la re-entrada) hasta superar max_age frames.
- AppUpdater: actualización de la apariencia del track asignado, con EMA (alpha) o con un gate aprendido (flag memory.app_update = ema | gate).
- Banco histórico de apariencias (opcional, h3.bank, appearance_bank.py): los últimos M embeddings FIABLES de cada track (bank (B,N,M,in_dim), el más reciente al final; bank_n válidos; bank_t frame
  de cada uno; last_write frame de la última escritura). Se escribe solo con una detección segura (score) y sin solape con otra (un cruce contamina el embedding), y como mucho cada write_every frames;
  durante una oclusión el track no se asigna, así que su banco se queda congelado con las apariencias de ANTES de la oclusión. Al nacer, el banco empieza con el embedding de su primera detección.
- EdgeMemory (fase 2, opcional con memory.edges): vector e_ij por pareja de huecos, actualizado por un GRU cuando ambos tracks se asignan en el frame; entra como sesgo
  adicional por head en la auto-atención de tracks (paso 3). Es un tensor (B,N,N,De) que se pone a 0 al reasignar un hueco (equivale al diccionario por pareja de IDs)."""
import torch
import torch.nn as nn

from tracking.e2e.h3.relational_attention import pair_geometry
from tracking.e2e.heads import mlp


class TrackMemory:
    def __init__(self, app, hist, hist_len, last, born, ids, alive, edges=None, bank=None, bank_n=None, bank_t=None, last_write=None):
        self.app, self.hist, self.hist_len, self.last, self.born, self.ids, self.alive, self.edges = app, hist, hist_len, last, born, ids, alive, edges
        self.bank, self.bank_n, self.bank_t, self.last_write = bank, bank_n, bank_t, last_write         # banco histórico de apariencias (None si está apagado)

    @classmethod
    def empty(cls, B, N, in_dim, K, device, edge_dim=0, bank_size=0):
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=device)
        return cls(z(B, N, in_dim), z(B, N, K, 4), z(B, N, dt=torch.long), z(B, N, dt=torch.long), z(B, N, dt=torch.long), -torch.ones(B, N, dtype=torch.long, device=device),
                   z(B, N, dt=torch.bool), z(B, N, N, edge_dim) if edge_dim > 0 else None,
                   z(B, N, bank_size, in_dim) if bank_size > 0 else None, z(B, N, dt=torch.long) if bank_size > 0 else None,
                   z(B, N, bank_size, dt=torch.long) if bank_size > 0 else None, z(B, N, dt=torch.long) if bank_size > 0 else None)

    @property
    def B(self): return self.app.shape[0]

    @property
    def N(self): return self.app.shape[1]

    def gap(self, frame):
        """frames desde la última asignación de cada track (frame: (B,) frame actual de cada clip)."""
        return (frame[:, None] - self.last).clamp(min=0)

    def age(self, frame):
        """frames desde el nacimiento de cada track."""
        return (frame[:, None] - self.born).clamp(min=0)

    def replace(self, **kw):
        d = dict(app=self.app, hist=self.hist, hist_len=self.hist_len, last=self.last, born=self.born, ids=self.ids, alive=self.alive, edges=self.edges,
                 bank=self.bank, bank_n=self.bank_n, bank_t=self.bank_t, last_write=self.last_write)
        d.update(kw)
        return TrackMemory(**d)

    def drop(self, dead):
        """Elimina los tracks marcados en dead (B,N)."""
        dead = dead & self.alive
        extra = {} if self.bank is None else dict(bank_n=torch.where(dead, torch.zeros_like(self.bank_n), self.bank_n))
        return self.replace(alive=self.alive & ~dead, ids=torch.where(dead, -torch.ones_like(self.ids), self.ids), hist_len=torch.where(dead, torch.zeros_like(self.hist_len), self.hist_len), **extra)

    def kill(self, frame, max_age):
        """Elimina los tracks con más de max_age frames sin asignar."""
        return self.drop(self.alive & (self.gap(frame) > max_age))

    def spawn(self, b, ids, app0, boxes, frame_b):
        """Crea tracks nuevos en los primeros huecos libres del clip b. ids (n,), app0 (n,in_dim), boxes (n,4). Devuelve (memoria, nº creados)."""
        free = (~self.alive[b]).nonzero().flatten()
        n = min(len(ids), len(free))
        if n == 0:
            return self, 0
        slot = free[:n]
        app, hist, hl, last, born, idv, alive = (x.clone() for x in (self.app, self.hist, self.hist_len, self.last, self.born, self.ids, self.alive))
        app[b, slot], idv[b, slot], alive[b, slot] = app0[:n], ids[:n], True
        hist[b, slot] = 0.0
        hist[b, slot, -1] = boxes[:n]
        hl[b, slot], last[b, slot], born[b, slot] = 1, frame_b, frame_b
        edges = None
        if self.edges is not None:       # el hueco empieza sin aristas (ni como origen ni como destino)
            edges = self.edges.clone()
            edges[b, slot], edges[b, :, slot] = 0.0, 0.0
        bank = bank_n = bank_t = last_write = None
        if self.bank is not None:        # el banco del hueco empieza con el embedding de su primera detección
            bank, bank_n, bank_t, last_write = self.bank.clone(), self.bank_n.clone(), self.bank_t.clone(), self.last_write.clone()
            bank[b, slot] = 0.0
            bank[b, slot, -1] = app0[:n]
            bank_n[b, slot], last_write[b, slot] = 1, frame_b
            bank_t[b, slot] = 0
            bank_t[b, slot, -1] = frame_b
        return TrackMemory(app, hist, hl, last, born, idv, alive, edges, bank, bank_n, bank_t, last_write), n

    def push(self, matched, box_obs, app_new, frame, emb_obs=None, write=None):
        """Tracks asignados (matched (B,N)): añaden su caja al historial, actualizan la apariencia y el último frame. Con banco, write (B,N) (⊆ matched) marca los que añaden emb_obs
        (B,N,in_dim), el embedding de su detección, al banco (el más antiguo sale)."""
        K = self.hist.shape[2]
        shifted = torch.cat([self.hist[:, :, 1:], box_obs[:, :, None]], 2)
        extra = {}
        if self.bank is not None and write is not None:
            M = self.bank.shape[2]
            stamp = frame[:, None].expand_as(self.last)
            extra = dict(bank=torch.where(write[..., None, None], torch.cat([self.bank[:, :, 1:], emb_obs[:, :, None]], 2), self.bank),
                         bank_t=torch.where(write[..., None], torch.cat([self.bank_t[:, :, 1:], stamp[..., None]], 2), self.bank_t),
                         bank_n=torch.where(write, (self.bank_n + 1).clamp(max=M), self.bank_n), last_write=torch.where(write, stamp, self.last_write))
        return self.replace(app=torch.where(matched[..., None], app_new, self.app), hist=torch.where(matched[..., None, None], shifted, self.hist),
                            hist_len=torch.where(matched, (self.hist_len + 1).clamp(max=K), self.hist_len),
                            last=torch.where(matched, frame[:, None].expand_as(self.last), self.last), **extra)


class AppUpdater(nn.Module):
    def __init__(self, in_dim=256, mode='ema', alpha=0.1):
        super().__init__()
        assert mode in ('ema', 'gate'), f'memory.app_update desconocido: {mode}'
        self.mode, self.alpha = mode, float(alpha)
        if mode == 'gate':
            self.gate = mlp(2 * in_dim, in_dim, in_dim)
            with torch.no_grad():
                self.gate[-1].bias.fill_(float(torch.logit(torch.tensor(1 - self.alpha))))     # arranca equivalente a la EMA

    def forward(self, old, obs):
        """old, obs (..., in_dim) -> nueva apariencia: g·old + (1-g)·obs, con g = 1 - alpha (ema) o aprendido por canal (gate)."""
        if self.mode == 'ema':
            return (1 - self.alpha) * old + self.alpha * obs
        g = torch.sigmoid(self.gate(torch.cat([old, obs], -1)))
        return g * old + (1 - g) * obs


class EdgeMemory(nn.Module):
    def __init__(self, edge_dim=32, heads=8):
        super().__init__()
        self.cell = nn.GRUCell(4, edge_dim)
        self.bias = nn.Linear(edge_dim, heads)

    def update(self, E, boxes, matched):
        """E (B,N,N,De), boxes (B,N,4) cajas asignadas en este frame, matched (B,N): GRU en las parejas con ambos tracks asignados."""
        B, N = matched.shape
        pair = matched[:, :, None] & matched[:, None, :] & ~torch.eye(N, dtype=torch.bool, device=E.device)
        geo = pair_geometry(boxes).clamp(-10, 10)
        new = self.cell(geo.flatten(0, 2), E.flatten(0, 2)).view_as(E)
        return torch.where(pair[..., None], new, E)

    def attn_bias(self, E):
        """(B,N,N,De) -> sesgo por head (B,H,N,N)."""
        return self.bias(E).permute(0, 3, 1, 2)
