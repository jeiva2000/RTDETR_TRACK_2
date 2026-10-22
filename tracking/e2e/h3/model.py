"""H3Model: asociación relacional sobre q_det + caja. Ensambla los componentes en el orden del esquema:
  1 AppearanceHead -> 2 MotionPredictor (GRU sobre el historial) -> 3 auto-atención de tracks con sesgo geométrico -> 4 auto-atención de detecciones ->
  5 atención cruzada detecciones <- tracks (s_rel = logits de la última capa, media entre heads) -> 6 gates por par -> 7 Sinkhorn con dustbin / Hungarian -> 8 memoria.
Una detección es una query con score > h3.det_thr (D variable, con relleno y máscara); los tracks son los huecos vivos de la memoria (T variable). Funciona con D = 0 y T = 0.
Flags (h3.branches.app/mot/rel, h3.gates, h3.memory.*) para las ablaciones: sin mot la caja predicha es la última observada y no hay GRU; sin rel no hay atención."""
import os, sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from tracking.e2e.h3.affinity_gates import AffinityGates
from tracking.e2e.h3.appearance_bank import HistoryPool, bank_affinity, bank_valid
from tracking.e2e.h3.appearance_head import AppearanceHead
from tracking.e2e.h3.birth_head import H3BirthHead
from tracking.e2e.h3.assignment import Assignment
from tracking.e2e.h3.geometry import giou_pairwise
from tracking.e2e.h3.motion_predictor import MotionPredictor
from tracking.e2e.h3.relational_attention import CrossLayer, GeoSelfLayer, NEG, pair_geometry
from tracking.e2e.h3.track_memory import AppUpdater, EdgeMemory, TrackMemory
from tracking.e2e.heads import box_iou_b


class H3Model(nn.Module):
    kind = 'h3'

    def __init__(self, cfg, in_dim=256):
        super().__init__()
        hc, br = cfg.h3, cfg.h3.branches
        self.cfg, self.in_dim = cfg, in_dim
        self.use_app, self.use_mot, self.use_rel = bool(br.app), bool(br.mot), bool(br.rel)
        dim, ad, H = hc.dim, hc.app_dim, hc.heads
        self.app = AppearanceHead(in_dim, in_dim, ad)
        self.bank_on = bool(hc.bank.enabled)
        self.hist_pool = HistoryPool(ad, hc.bank.heads) if self.bank_on else None                     # el token del track consulta su propio banco de apariencias
        self.motion = MotionPredictor(hc.gru_dim, ad, hc.gru_dim, hc.delta_scale) if self.use_mot else None
        if self.use_rel:
            self.trk_in, self.box_embed, self.det_in = nn.Linear((3 if hc.bank.enabled else 2) * ad, dim), nn.Linear(4, ad), nn.Linear(2 * ad, dim)    # token del track: [apariencia, movimiento (, resumen del banco)]
            self.trk_layers = nn.ModuleList(GeoSelfLayer(dim, H, hc.ffn, 6, hc.geo_hidden) for _ in range(hc.layers_trk))
            self.det_layers = nn.ModuleList(GeoSelfLayer(dim, H, hc.ffn, 4, hc.geo_hidden) for _ in range(hc.layers_det))
            self.cross_layers = nn.ModuleList(CrossLayer(dim, H, hc.ffn) for _ in range(hc.layers_x))
        self.gates = AffinityGates((self.use_app, self.use_mot, self.use_rel), bool(hc.gates), tuple(hc.fixed_w), hc.tau_init)
        self.assign = Assignment(hc.sinkhorn_iters)
        self.updater = AppUpdater(in_dim, hc.memory.app_update, hc.memory.alpha)
        self.edges = EdgeMemory(hc.memory.edge_dim, H) if (hc.memory.edges and self.use_rel) else None
        self.birth = H3BirthHead(ad, hc.birth.hidden) if hc.birth.enabled else None            # nacimientos aprendidos; sin cabeza, regla por score (h3.new_thr)

    @property
    def hc(self):
        """Configuración h3 ACTUAL: se lee de cfg en cada uso (cfg.h3 devuelve una copia, así que guardarla en __init__ dejaría los umbrales fijos y los cambios posteriores,
        como los del barrido de sweep.py (h3.min_prob, h3.new_thr), no llegarían al modelo)."""
        return self.cfg.h3

    def new_memory(self, B, device):
        return TrackMemory.empty(B, self.cfg.state.max_tracks, self.in_dim, self.hc.K_hist, device, self.hc.memory.edge_dim if self.edges is not None else 0,
                                 self.hc.bank.size if self.bank_on else 0)

    def det_valid(self, dets):
        return dets.valid & (dets.score > self.hc.det_thr)

    def init_app(self, emb):
        """Apariencia inicial de un track nuevo = el embedding de su primera detección (en el espacio del decoder)."""
        return emb

    def forward_scores(self, mem, dets, frame):
        """Pasos 1-7 (sin la decisión). -> dict con a_det, a_trk, box_pred, vel, s_app, s_mot, s_rel, gates, s (B,D,T), logP (B,D+1,T+1), det_valid, trk_valid."""
        dv, tv = self.det_valid(dets), mem.alive
        B, D, T = dets.emb.shape[0], dets.emb.shape[1], mem.N
        a_det, a_trk = self.app(dets.emb), self.app(mem.app)
        gap = mem.gap(frame)
        if self.motion is not None:
            box_pred, vel, m = self.motion(mem.hist, mem.hist_len, gap)
        else:
            box_pred, vel, m = mem.hist[:, :, -1], mem.app.new_zeros(B, T, 4), mem.app.new_zeros(B, T, self.hc.app_dim)
        zero = a_det.new_zeros(B, D, T)
        h_hist = None
        if self.bank_on:                                                                     # banco histórico: afinidad con el máximo sobre el banco y resumen del historial para el token
            M = mem.bank.shape[2]
            a_bank, bv = self.app(mem.bank), bank_valid(mem.bank_n, M)
            s_app, s_ema = bank_affinity(a_det, a_trk, a_bank, bv, self.hc.bank.affinity)
            h_hist = self.hist_pool(a_trk, a_bank, (frame[:, None, None] - mem.bank_t).clamp(min=0), bv)
        else:
            s_app = s_ema = a_det @ a_trk.transpose(1, 2)
        s_app = s_app if self.use_app else zero
        s_mot = giou_pairwise(dets.box, box_pred) if self.use_mot else zero
        s_rel = zero
        if self.use_rel:
            xt = self.trk_in(torch.cat([a_trk, m] + ([h_hist] if h_hist is not None else []), -1))
            geo_t = pair_geometry(box_pred, vel[..., :2])                                    # geometría de los tracks con la caja PREDICHA en t
            eb = self.edges.attn_bias(mem.edges) if self.edges is not None else None
            for layer in self.trk_layers:
                xt = layer(xt, tv, geo_t, eb)
            xd = self.det_in(torch.cat([a_det, self.box_embed(dets.box)], -1))
            geo_d = pair_geometry(dets.box)
            for layer in self.det_layers:
                xd = layer(xd, dv, geo_d)
            for layer in self.cross_layers:
                xd, logits = layer(xd, xt, tv)
            s_rel = logits.mean(1)                                                           # (B,D,T): logits de la última capa, media entre heads
        s, g = self.gates(s_app, s_mot, s_rel, tv)
        logP = self.assign(s, dv, tv)
        return dict(a_det=a_det, a_trk=a_trk, box_pred=box_pred, vel=vel, s_app=s_app, s_app_ema=s_ema, s_mot=s_mot, s_rel=s_rel, gates=g, s=s, logP=logP, det_valid=dv, trk_valid=tv,
                    birth_logit=self.birth_logit(a_det, dets, box_pred, s, logP, tv))

    def birth_logit(self, a_det, dets, box_pred, s, logP, tv):
        """Logit (B,D) de que cada detección deba crear un track nuevo (None sin cabeza). Entradas de birth_head.py; con h3.birth.detach (por defecto) no hay gradiente hacia la asociación."""
        if self.birth is None:
            return None
        B, D, T = s.shape
        P = logP[:, :D, :].exp()
        p_bin = P[..., T]
        if T > 0:
            alive = tv[:, None, :]
            claim = P[..., :T].masked_fill(~alive, 0.0).max(2).values
            iou_max = box_iou_b(dets.box, box_pred).masked_fill(~alive, 0.0).max(2).values
            s_max = torch.where(tv.any(1)[:, None], s.masked_fill(~alive, NEG).max(2).values, torch.zeros_like(p_bin))
        else:
            claim = iou_max = s_max = torch.zeros_like(p_bin)
        n_alive = (tv.sum(1).float() / 10)[:, None].expand(B, D)
        feats = [dets.score, p_bin, claim, iou_max, s_max / 10, n_alive]
        if self.hc.birth.detach:
            a_det, feats = a_det.detach(), [f.detach() for f in feats]
        return self.birth(a_det, *feats)

    def decide(self, out, target=None, use_model=None):
        """Asignación (B,T): detección de cada track o D (no observado). Hungarian sobre s + umbral de probabilidad h3.min_prob, o el target (teacher forcing)."""
        return self.assign.decide(out['s'], out['logP'], out['det_valid'], out['trk_valid'], target, use_model, self.hc.min_prob)

    def update(self, mem, assign, dets, frame):
        """Paso 8: los tracks asignados actualizan apariencia (EMA o gate), historial y último frame. -> (memoria, matched (B,N))."""
        D = dets.emb.shape[1]
        matched = (assign < D) & mem.alive
        if D == 0:
            return mem, matched
        j = assign.clamp(max=D - 1)
        emb_obs = torch.gather(dets.emb, 1, j[..., None].expand(-1, -1, dets.emb.shape[-1]))
        box_obs = torch.gather(dets.box, 1, j[..., None].expand(-1, -1, 4))
        write = None
        if self.bank_on:                                                                     # escribir en el banco solo con detecciones fiables: score alto, sin solape con otra y no demasiado seguido
            bk = self.hc.bank
            dv = self.det_valid(dets)
            ov = box_iou_b(dets.box, dets.box).masked_fill(~dv[:, None, :] | torch.eye(D, dtype=torch.bool, device=dets.box.device)[None], 0.0).max(2).values       # mayor IoU con otra detección válida
            gather = lambda x: torch.gather(x, 1, j)
            write = matched & (gather(dets.score) >= bk.write_conf) & (gather(ov) < bk.write_overlap) & ((frame[:, None] - mem.last_write) >= bk.write_every)
        new = mem.push(matched, box_obs, self.updater(mem.app, emb_obs), frame, emb_obs, write)
        if self.edges is not None:
            new = new.replace(edges=self.edges.update(mem.edges, box_obs, matched))
        return new, matched
