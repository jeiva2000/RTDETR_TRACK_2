"""TrackHeads (etapa 1). Módulo de seguimiento sobre un RT-DETRv2 congelado: cada detección del frame t recibe su propia cabeza de atención
(generada por una hipernetwork a partir de su query y su caja), que localiza el mismo objeto en la memoria del encoder del frame t+k y predice
si sigue presente.

Notación: B lote, T tracks (con relleno y track_mask), N tokens de la memoria aplanada, D = 256, d_h = 64, r = 8. Cajas en cxcywh normalizado.

1. Memoria: los 3 niveles del encoder (strides 8/16/32) se aplanan y concatenan -> mem (B,N,D); xy (N,2) es el centro normalizado de cada token;
   x = mem + Linear(seno2D(xy)) + level_embed[nivel].
2. HeadGenerator: z_i = [emb_i, prev_box_i] (D+4) -> q_i = MLP(z_i) (d_h), U_i (D×r, escalada por 1/√D) y V_i (d_h×r, inicializada a cero).
   ΔW_i = U_i V_iᵀ es una corrección de rango r, propia del track, a la proyección de valores W_v.
3. Atención: K = W_k x (W_k compartida: con una sola query por cabeza, una W_k propia se absorbería en la query).
   logits_i = q_i·K/√d_h − ‖xy − c_i‖²/(2 s_i²) (prior gaussiano opcional), s_i = max(exp(log_scale)·max(w_i,h_i), s_min); softmax sobre N.
4. Lectura: x̄_i = attn_i·mem; v_i = W_v x̄_i + (x̄_i U_i) V_iᵀ; shift_i = attn_i·xy − c_i.
   Nunca se materializa un tensor (B,T,N,d_h): la atención es un producto (B,T,d_h)×(B,N,d_h) y la corrección se aplica tras promediar.
5. Salida: MLP de 3 capas sobre [v, emb, shift] -> (δx, δy, δw, δh, logit de presencia), última capa a cero (al inicio la caja es prev_box
   y la presencia 0.5). Caja: cx = cx0 + δx·w0, cy = cy0 + δy·h0, w = w0·exp(clamp(δw,±3)), h = h0·exp(clamp(δh,±3)).

6. Asociación (Associator, TrackHeads.associate): puntúa cada detección candidata j del frame t+k para el track i,
   s_ij = escala·cos(h_i, c_j) + w_masa·log(masa de attn_i dentro de la caja j) + w_iou·IoU(caja predicha_i, caja j),
   con h_i = MLP([v_i, emb_i]) y c_j = MLP([emb_j, caja_j]) (query y caja del detector en t+k), más una clase "ninguna" (MLP([v_i, emb_i])) para
   tracks cuya identidad no está entre las candidatas (ausente o no detectada). Se entrena con entropía cruzada: castiga directamente apuntar
   a la persona de al lado. w_iou empieza en 5 (al inicio la caja predicha es prev_box: arranca como un tracker por IoU) y w_masa en 1.

Los tracks de relleno (track_mask = False) se calculan igual (valores finitos, independientes del resto: cada cabeza mira solo la memoria)
y se devuelven sin tocar; las pérdidas deben enmascararlos con track_mask."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tracking.e2e.heads import mlp


def sine_2d(xy, dim=256, temperature=10000.0):
    """Codificación senoidal 2D de coordenadas continuas en [0,1]. xy (...,2) -> (...,dim): [sin/cos(x), sin/cos(y)], dim/2 por eje."""
    assert dim % 4 == 0, 'dim debe ser múltiplo de 4'
    npf = dim // 2
    i = torch.arange(npf, dtype=xy.dtype, device=xy.device)
    dim_t = temperature ** (2 * torch.div(i, 2, rounding_mode='floor') / npf)
    p = xy[..., None] * (2 * math.pi) / dim_t                                       # (...,2,npf)
    p = torch.stack([p[..., 0::2].sin(), p[..., 1::2].cos()], -1).flatten(-2)       # (...,2,npf) intercalado sin/cos
    return p.flatten(-2)                                                            # (...,dim): x primero, luego y


def token_centers(shapes, device=None, dtype=torch.float32):
    """Centros normalizados de los tokens de cada nivel, en el orden de feat.flatten(2) (fila a fila).
    shapes [(H,W), ...] -> xy (N,2) en (x, y) y level (N,) con el índice del nivel."""
    xys, lvls = [], []
    for l, (H, W) in enumerate(shapes):
        ys = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H
        xs = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        xys.append(torch.stack([gx.flatten(), gy.flatten()], -1))
        lvls.append(torch.full((H * W,), l, device=device, dtype=torch.long))
    return torch.cat(xys), torch.cat(lvls)


class HeadGenerator(nn.Module):
    """Hipernetwork: z_i = [emb_i, prev_box_i] -> query propia q_i (d_h) y factores U_i (D×r), V_i (d_h×r) de la corrección de rango r."""

    def __init__(self, dim=256, d_h=64, rank=8, hidden=256):
        super().__init__()
        self.dim, self.d_h, self.rank = dim, d_h, rank
        z = dim + 4
        self.q = mlp(z, hidden, d_h)
        self.u = nn.Linear(z, dim * rank)
        self.v = nn.Linear(z, d_h * rank)
        nn.init.zeros_(self.v.weight)
        nn.init.zeros_(self.v.bias)

    def forward(self, emb, prev_box):
        """emb (B,T,D), prev_box (B,T,4) -> q (B,T,d_h), U (B,T,D,r), V (B,T,d_h,r)."""
        z = torch.cat([emb, prev_box], -1)
        B, T = z.shape[:2]
        U = self.u(z).view(B, T, self.dim, self.rank) / math.sqrt(self.dim)
        V = self.v(z).view(B, T, self.d_h, self.rank)
        return self.q(z), U, V


def tokens_in_boxes(token_xy, boxes):
    """token_xy (N,2), boxes (B,C,4) cxcywh -> (B,C,N) bool: el centro del token cae dentro de la caja."""
    x, y = token_xy[:, 0], token_xy[:, 1]
    return ((x - boxes[..., 0, None]).abs() <= boxes[..., 2, None] / 2) & ((y - boxes[..., 1, None]).abs() <= boxes[..., 3, None] / 2)


def pair_iou(a, b):
    """a (B,T,4), b (B,C,4) cxcywh -> IoU (B,T,C), diferenciable."""
    a, b = a[:, :, None], b[:, None]
    ix = (torch.min(a[..., 0] + a[..., 2] / 2, b[..., 0] + b[..., 2] / 2) - torch.max(a[..., 0] - a[..., 2] / 2, b[..., 0] - b[..., 2] / 2)).clamp(min=0)
    iy = (torch.min(a[..., 1] + a[..., 3] / 2, b[..., 1] + b[..., 3] / 2) - torch.max(a[..., 1] - a[..., 3] / 2, b[..., 1] - b[..., 3] / 2)).clamp(min=0)
    inter = ix * iy
    return inter / (a[..., 2] * a[..., 3] + b[..., 2] * b[..., 3] - inter + 1e-9)


NEG = -1e4      # puntuación de las detecciones que no son candidatas (finita: segura con amp y con la entropía cruzada)


class Associator(nn.Module):
    """Puntuación track × detección candidata de t+k (apariencia + atención + geometría) y clase "ninguna"."""

    def __init__(self, dim=256, d_h=64, d_a=128, hidden=256, app_scale=10.0, w_mass=1.0, w_iou=5.0):
        super().__init__()
        self.track = mlp(d_h + dim, hidden, d_a)
        self.cand = mlp(dim + 4, hidden, d_a)
        self.null = mlp(d_h + dim, hidden, 1)
        self.log_app = nn.Parameter(torch.tensor(math.log(app_scale)))
        self.w_mass = nn.Parameter(torch.tensor(float(w_mass)))
        self.w_iou = nn.Parameter(torch.tensor(float(w_iou)))

    def forward(self, feat, emb, boxes, attn, token_xy, emb_k, box_k, cand):
        """feat (B,T,d_h) lectura v, emb (B,T,D), boxes (B,T,4) predichas, attn (B,T,N); emb_k (B,C,D), box_k (B,C,4), cand (B,C) bool.
        -> logits (B,T,C+1) (la última columna es "ninguna") y sus términos: app, log_mass, iou (B,T,C)."""
        z = torch.cat([feat, emb], -1)
        h = F.normalize(self.track(z), dim=-1)
        c = F.normalize(self.cand(torch.cat([emb_k, box_k], -1)), dim=-1)
        app = self.log_app.exp() * torch.einsum('bta,bca->btc', h, c)
        mass = torch.bmm(attn, tokens_in_boxes(token_xy, box_k).transpose(1, 2).to(attn.dtype))
        log_mass = (mass + 1e-4).log()
        iou = pair_iou(boxes, box_k)
        s = (app + self.w_mass * log_mass + self.w_iou * iou).masked_fill(~cand[:, None], NEG)
        return torch.cat([s, self.null(z)], -1), dict(app=app.masked_fill(~cand[:, None], NEG), log_mass=log_mass, iou=iou)


class TrackHeads(nn.Module):
    def __init__(self, dim=256, d_h=64, rank=8, n_levels=3, hidden=256, gen_hidden=256,
                 use_prior=True, scale_init=2.0, min_scale=0.02, max_log_wh=3.0):
        super().__init__()
        self.dim, self.d_h, self.n_levels = dim, d_h, n_levels
        self.use_prior, self.min_scale, self.max_log_wh = use_prior, float(min_scale), float(max_log_wh)
        self.pos_proj = nn.Linear(dim, dim)
        self.level_embed = nn.Parameter(torch.empty(n_levels, dim))
        nn.init.normal_(self.level_embed, std=0.02)
        self.gen = HeadGenerator(dim, d_h, rank, gen_hidden)
        self.w_k = nn.Linear(dim, d_h)
        self.w_v = nn.Linear(dim, d_h)
        self.log_scale = nn.Parameter(torch.tensor(math.log(scale_init)))
        self.out = nn.Sequential(nn.Linear(d_h + dim + 2, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, 5))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)
        self.assoc = Associator(dim, d_h, hidden=hidden)

    # 1. memoria
    def memory(self, feats):
        """feats: lista de n_levels mapas (B,D,H,W) -> mem (B,N,D) sin posición, x (B,N,D) con posición y nivel, xy (N,2)."""
        assert len(feats) == self.n_levels, f'se esperaban {self.n_levels} niveles, llegan {len(feats)}'
        mem = torch.cat([f.flatten(2).transpose(1, 2) for f in feats], 1)
        xy, lvl = token_centers([f.shape[-2:] for f in feats], mem.device, mem.dtype)
        x = mem + self.pos_proj(sine_2d(xy, self.dim))[None] + self.level_embed[lvl][None]
        return mem, x, xy

    def scale(self, prev_box):
        """s_i = max(exp(log_scale)·max(w_i,h_i), min_scale) -> (B,T)."""
        return (self.log_scale.exp() * prev_box[..., 2:].amax(-1)).clamp(min=self.min_scale)

    def decode(self, prev_box, delta):
        """Caja relativa a prev_box (cx0, cy0, w0, h0) con delta (B,T,4) -> (B,T,4)."""
        cx0, cy0, w0, h0 = prev_box.unbind(-1)
        dx, dy, dw, dh = delta.unbind(-1)
        m = self.max_log_wh
        return torch.stack([cx0 + dx * w0, cy0 + dy * h0, w0 * dw.clamp(-m, m).exp(), h0 * dh.clamp(-m, m).exp()], -1)

    def forward(self, emb, prev_box, feats, track_mask):
        """emb (B,T,D), prev_box (B,T,4), feats [3×(B,D,H,W)] del frame t+k, track_mask (B,T) ->
        dict(boxes (B,T,4), presence_logits (B,T), attn (B,T,N), token_xy (N,2), track_mask (B,T))."""
        mem, x, xy = self.memory(feats)
        q, U, V = self.gen(emb, prev_box)

        # 3. atención: una query por track contra K compartida; prior gaussiano centrado en la caja anterior
        K = self.w_k(x)                                                              # (B,N,d_h)
        logits = torch.einsum('btd,bnd->btn', q, K) / math.sqrt(self.d_h)
        c = prev_box[..., :2]
        if self.use_prior:
            d2 = (xy[None, None] - c[:, :, None]).pow(2).sum(-1)                     # (B,T,N)
            logits = logits - d2 / (2 * self.scale(prev_box).pow(2)[..., None])
        attn = logits.softmax(-1)

        # 4. lectura propia: promedio de la memoria y corrección de rango r después de promediar (lineal, equivale a hacerlo por token)
        xbar = torch.bmm(attn, mem)                                                  # (B,T,D)
        v = self.w_v(xbar) + torch.einsum('btr,bthr->bth', torch.einsum('btd,btdr->btr', xbar, U), V)
        shift = attn @ xy - c                                                         # (B,T,2)

        # 5. salida
        o = self.out(torch.cat([v, emb, shift], -1))
        return {'boxes': self.decode(prev_box, o[..., :4]), 'presence_logits': o[..., 4], 'attn': attn,
                'token_xy': xy, 'track_mask': track_mask, 'feat': v, 'emb': emb}

    def associate(self, out, emb_k, box_k, cand):
        """6. Añade a out las puntuaciones de asociación contra las detecciones de t+k: emb_k (B,C,D), box_k (B,C,4), cand (B,C) bool
        (candidatas: score >= det_thr). -> out con assoc_logits (B,T,C+1; la última columna es "ninguna") y assoc_app (B,T,C)."""
        logits, parts = self.assoc(out['feat'], out['emb'], out['boxes'], out['attn'], out['token_xy'], emb_k, box_k, cand)
        out['assoc_logits'], out['assoc_app'] = logits, parts['app']
        return out
