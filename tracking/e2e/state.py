"""Estado de los tracks en tensores con relleno y máscaras: (B clips, N huecos máximos). Las operaciones son fuera de lugar
(devuelven un estado nuevo) para ser seguras con autograd/BPTT. `alive` marca los huecos ocupados; ids=-1 en los libres.
`m` (B,N,M) es el vector de movimiento del H2 (estado del GRU); None cuando motion.mode=none (H1)."""
import torch


class TrackState:
    def __init__(self, a, box, last, ids, alive, m=None):
        self.a, self.box, self.last, self.ids, self.alive = a, box, last, ids, alive   # (B,N,D) (B,N,4) (B,N) (B,N) (B,N)
        self.m = m                                                                       # (B,N,M) o None

    @classmethod
    def empty(cls, B, N, D, device, M=0):
        return cls(torch.zeros(B, N, D, device=device), torch.zeros(B, N, 4, device=device),
                   torch.zeros(B, N, dtype=torch.long, device=device), -torch.ones(B, N, dtype=torch.long, device=device),
                   torch.zeros(B, N, dtype=torch.bool, device=device), torch.zeros(B, N, M, device=device) if M > 0 else None)

    @property
    def B(self): return self.a.shape[0]

    @property
    def N(self): return self.a.shape[1]

    def n_alive(self): return self.alive.sum(1)

    def gap(self, frame):
        """frames desde que se vio cada track (frame: (B,) frame actual de cada clip)."""
        return (frame[:, None] - self.last).clamp(min=0)

    def replace(self, **kw):
        d = dict(a=self.a, box=self.box, last=self.last, ids=self.ids, alive=self.alive, m=self.m)
        d.update(kw)
        return TrackState(**d)

    def kill(self, frame, max_age):
        """Elimina los tracks con más de max_age frames sin verse."""
        dead = self.alive & (self.gap(frame) > max_age)
        return self.replace(alive=self.alive & ~dead, ids=torch.where(dead, -torch.ones_like(self.ids), self.ids))

    def spawn(self, b, ids, a0, boxes, frame_b):
        """Crea tracks nuevos en los primeros huecos libres del clip b. ids (n,), a0 (n,D), boxes (n,4). Devuelve (estado, n_creados)."""
        free = (~self.alive[b]).nonzero().flatten()
        n = min(len(ids), len(free))
        if n == 0:
            return self, 0
        slot = free[:n]
        a, box, last, idv, alive = (x.clone() for x in (self.a, self.box, self.last, self.ids, self.alive))
        a[b, slot], box[b, slot], idv[b, slot] = a0[:n], boxes[:n], ids[:n]
        last[b, slot], alive[b, slot] = frame_b, True
        m = None
        if self.m is not None:       # un track nuevo no tiene historia de movimiento: m = 0
            m = self.m.clone()
            m[b, slot] = 0.0
        return TrackState(a, box, last, idv, alive, m), n
