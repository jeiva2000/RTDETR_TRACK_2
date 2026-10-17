"""Módulo de asociación sobre las queries del detector congelado.

Interfaz común: cada track tiene un estado s (dim D) que nace de su primera detección y se actualiza en cada frame.
- Afinidad: logits (N, K+1) = MLP_app([s_i, e_j, s_i*e_j]) + MLP_geo(geo_ij) para las K detecciones del frame
  (dos ramas sumadas para que la geometría no quede enterrada entre 768 dims de apariencia), más un token "no aparece"
  --app-aff cos: la rama de apariencia es una similitud coseno entre proyecciones con temperatura aprendida
  (solo puede medir parecido; la MLP sobre la concatenación tiende a memorizar identidades de train).
  por track; softmax por fila y entropía cruzada contra el emparejamiento GT.
- update: cinco variantes con todo lo demás idéntico:
    none  : s no cambia (solo la apariencia inicial)
    ema   : s <- a*s + (1-a)*enc(e_obs)
    gru   : s <- g*GRU(enc(e_obs), s) + (1-g)*s,  g = sigmoid(MLP([confianza del match, solapamiento con otros tracks]))
    graph : como gru, pero la entrada de la GRU es [enc(e_obs), m_i], con m_i = message passing atencional de las
            detecciones del frame cercanas al track (aristas filtradas por distancia espacial). Los tracks no
            observados también se actualizan solo con m_i (confianza 0 en la puerta).
    graph2: como graph, y además un segundo mensaje m'_i de los OTROS tracks cercanos (atención sobre sus estados s_j y su
            geometría relativa: desplazamiento, escala, IoU y tiempo sin verse; usa las cajas ya actualizadas de este frame),
            de modo que cada track sabe qué vecinos compiten por la misma zona. La GRU recibe [enc(e_obs), m_i, m'_i].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

GEO_DIM = 7
TRK_GEO_DIM = 6
UPDATES = ('none', 'ema', 'gru', 'graph', 'graph2')


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(inplace=True), nn.Linear(h, o))


def box_iou_cxcywh(a, b):
    """a (N,4) b (K,4) cxcywh -> (N,K)"""
    ax1, ay1, ax2, ay2 = a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2, a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2
    bx1, by1, bx2, by2 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    iw = (torch.min(ax2[:, None], bx2[None]) - torch.max(ax1[:, None], bx1[None])).clamp(min=0)
    ih = (torch.min(ay2[:, None], by2[None]) - torch.max(ay1[:, None], by1[None])).clamp(min=0)
    inter = iw * ih
    return inter / (a[:, 2:3] * a[:, 3:4] + (b[:, 2] * b[:, 3])[None] - inter + 1e-9)


def geometry(tb, gap, db, scores):
    """Rasgos geométricos track->detección (N,K,7): dx/w, dy/h, log(w'/w), log(h'/h), IoU, score, frames sin ver/10."""
    w, h = tb[:, 2:3].clamp(min=1e-4), tb[:, 3:4].clamp(min=1e-4)
    dx = (db[None, :, 0] - tb[:, None, 0]) / w
    dy = (db[None, :, 1] - tb[:, None, 1]) / h
    lw = torch.log(db[None, :, 2].clamp(min=1e-4) / w)
    lh = torch.log(db[None, :, 3].clamp(min=1e-4) / h)
    iou = box_iou_cxcywh(tb, db)
    N, K = iou.shape
    return torch.stack([dx, dy, lw, lh, iou, scores[None].expand(N, K), (gap.float() / 10)[:, None].expand(N, K)], -1)


class AssocModel(nn.Module):
    def __init__(self, update='gru', in_dim=256, dim=256, hidden=256, ema_alpha=0.9, graph_radius=2.0, app_aff='mlp'):
        super().__init__()
        assert update in UPDATES, update
        self.update_type, self.ema_alpha, self.graph_radius = update, ema_alpha, graph_radius
        self.proj = nn.Sequential(nn.Linear(in_dim, dim), nn.LayerNorm(dim))
        self.enc = mlp(dim, hidden, dim)                       # enc(e): espacio del estado
        assert app_aff in ('mlp', 'cos'), app_aff
        self.app_aff = app_aff
        if app_aff == 'mlp':
            self.aff = mlp(3 * dim, hidden, 1)
        else:
            self.q_s, self.q_p = nn.Linear(dim, dim), nn.Linear(dim, dim)
            self.log_tau = nn.Parameter(torch.tensor(2.3))  # exp(2.3) ~ 10
        self.aff_geo = mlp(GEO_DIM, 64, 1)
        self.absent = mlp(dim + 1, hidden, 1)                  # token "no aparece": [s, frames sin ver/10]
        if update in ('gru', 'graph', 'graph2'):
            self.gate = mlp(2, 32, 1)
            self.gru = nn.GRUCell(dim * {'gru': 1, 'graph': 2, 'graph2': 3}[update], dim)
        if update in ('graph', 'graph2'):
            self.msg = mlp(2 * dim + GEO_DIM, hidden, dim)
            self.att = mlp(2 * dim + GEO_DIM, hidden // 4, 1)
        if update == 'graph2':
            self.msg_t = mlp(2 * dim + TRK_GEO_DIM, hidden, dim)
            self.att_t = mlp(2 * dim + TRK_GEO_DIM, hidden // 4, 1)

    def encode(self, e):
        p = self.proj(e)
        return p, self.enc(p)

    def init_state(self, e):
        """Estado inicial a partir del embedding de la query de la primera detección."""
        return self.encode(e)[1]

    def affinity(self, S, tb, gap, E, D, sc, valid=None):
        """S (N,dim) tb (N,4) gap (N,) | E (K,256) D (K,4) sc (K,) valid (K,) bool -> logits (N,K+1), geo, p"""
        p, _ = self.encode(E)
        N, K = S.shape[0], E.shape[0]
        geo = geometry(tb, gap, D, sc)
        if self.app_aff == 'mlp':
            x = torch.cat([S[:, None].expand(N, K, -1), p[None].expand(N, K, -1), S[:, None] * p[None]], -1)
            app = self.aff(x).squeeze(-1)
        else:
            app = self.log_tau.exp() * F.normalize(self.q_s(S), dim=-1) @ F.normalize(self.q_p(p), dim=-1).T
        logit = app + self.aff_geo(geo).squeeze(-1)
        if valid is not None:
            logit = logit.masked_fill(~valid[None], float('-inf'))
        absent = self.absent(torch.cat([S, (gap.float() / 10)[:, None]], -1))
        return torch.cat([logit, absent], 1), geo, p

    def update(self, S, idx, conf, overlap, E, geo, p, tb=None, gap=None):
        """idx (N,) índice de la detección asignada (-1 = no aparece); conf, overlap (N,) en [0,1].
        tb (N,4) cajas de los tracks tras la asignación y gap (N,) frames sin verse tras ella (solo los usa graph2)."""
        obs = idx >= 0
        if self.update_type == 'none':
            return S
        enc_obs = torch.zeros_like(S)
        if obs.any():
            enc_obs[obs] = self.encode(E[idx[obs]])[1]
        if self.update_type == 'ema':
            a = self.ema_alpha
            return torch.where(obs[:, None], a * S + (1 - a) * enc_obs, S)
        g = torch.sigmoid(self.gate(torch.stack([conf, overlap], -1)))
        if self.update_type == 'gru':
            h = self.gru(enc_obs, S)
            return torch.where(obs[:, None], g * h + (1 - g) * S, S)
        # graph: mensajes de las detecciones cercanas (|dx|,|dy| < radio en unidades del tamaño del track)
        N, K = geo.shape[:2]
        near = (geo[..., 0].abs() < self.graph_radius) & (geo[..., 1].abs() < self.graph_radius)
        x = torch.cat([S[:, None].expand(N, K, -1), p[None].expand(N, K, -1), geo], -1)
        a = self.att(x).squeeze(-1).masked_fill(~near, float('-inf'))
        w = torch.nan_to_num(torch.softmax(a, 1), nan=0.0)   # tracks sin vecinos -> mensaje 0
        m = (w[..., None] * self.msg(x)).sum(1)
        if self.update_type == 'graph2':
            m2 = self.track_message(S, tb, gap)
            h = self.gru(torch.cat([enc_obs, m, m2], -1), S)
        else:
            h = self.gru(torch.cat([enc_obs, m], -1), S)
        return g * h + (1 - g) * S

    def track_message(self, S, tb, gap):
        """Mensaje de los otros tracks cercanos (aristas j->i si |dx|,|dy| < radio, en unidades del tamaño de i; sin autoaristas)."""
        N = S.shape[0]
        if N < 2:
            return torch.zeros_like(S)
        w, h = tb[:, 2:3].clamp(min=1e-4), tb[:, 3:4].clamp(min=1e-4)            # (N,1) tamaño del receptor i
        dx = (tb[None, :, 0] - tb[:, None, 0]) / w                                 # (i,j)
        dy = (tb[None, :, 1] - tb[:, None, 1]) / h
        lw = torch.log(tb[None, :, 2].clamp(min=1e-4) / w)
        lh = torch.log(tb[None, :, 3].clamp(min=1e-4) / h)
        iou = box_iou_cxcywh(tb, tb)
        gj = (gap.float() / 10)[None, :].expand(N, N)
        tg = torch.stack([dx, dy, lw, lh, iou, gj], -1)
        near = (dx.abs() < self.graph_radius) & (dy.abs() < self.graph_radius) & ~torch.eye(N, dtype=torch.bool, device=S.device)
        x = torch.cat([S[:, None].expand(N, N, -1), S[None].expand(N, N, -1), tg], -1)
        a = self.att_t(x).squeeze(-1).masked_fill(~near, float('-inf'))
        wgt = torch.nan_to_num(torch.softmax(a, 1), nan=0.0)                       # sin vecinos -> mensaje 0
        return (wgt[..., None] * self.msg_t(x)).sum(1)
