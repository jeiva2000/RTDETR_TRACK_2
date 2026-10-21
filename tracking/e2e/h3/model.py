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
from tracking.e2e.h3.appearance_head import AppearanceHead
from tracking.e2e.h3.assignment import Assignment
from tracking.e2e.h3.geometry import giou_pairwise
from tracking.e2e.h3.motion_predictor import MotionPredictor
from tracking.e2e.h3.relational_attention import CrossLayer, GeoSelfLayer, pair_geometry
from tracking.e2e.h3.track_memory import AppUpdater, EdgeMemory, TrackMemory


class H3Model(nn.Module):
    kind = 'h3'

    def __init__(self, cfg, in_dim=256):
        super().__init__()
        hc, br = cfg.h3, cfg.h3.branches
        self.cfg, self.hc, self.in_dim = cfg, hc, in_dim
        self.use_app, self.use_mot, self.use_rel = bool(br.app), bool(br.mot), bool(br.rel)
        dim, ad, H = hc.dim, hc.app_dim, hc.heads
        self.app = AppearanceHead(in_dim, in_dim, ad)
        self.motion = MotionPredictor(hc.gru_dim, ad, hc.gru_dim, hc.delta_scale) if self.use_mot else None
        if self.use_rel:
            self.trk_in, self.box_embed, self.det_in = nn.Linear(2 * ad, dim), nn.Linear(4, ad), nn.Linear(2 * ad, dim)
            self.trk_layers = nn.ModuleList(GeoSelfLayer(dim, H, hc.ffn, 6, hc.geo_hidden) for _ in range(hc.layers_trk))
            self.det_layers = nn.ModuleList(GeoSelfLayer(dim, H, hc.ffn, 4, hc.geo_hidden) for _ in range(hc.layers_det))
            self.cross_layers = nn.ModuleList(CrossLayer(dim, H, hc.ffn) for _ in range(hc.layers_x))
        self.gates = AffinityGates((self.use_app, self.use_mot, self.use_rel), bool(hc.gates), tuple(hc.fixed_w), hc.tau_init)
        self.assign = Assignment(hc.sinkhorn_iters)
        self.updater = AppUpdater(in_dim, hc.memory.app_update, hc.memory.alpha)
        self.edges = EdgeMemory(hc.memory.edge_dim, H) if (hc.memory.edges and self.use_rel) else None

    def new_memory(self, B, device):
        return TrackMemory.empty(B, self.cfg.state.max_tracks, self.in_dim, self.hc.K_hist, device, self.hc.memory.edge_dim if self.edges is not None else 0)

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
        s_app = a_det @ a_trk.transpose(1, 2) if self.use_app else zero
        s_mot = giou_pairwise(dets.box, box_pred) if self.use_mot else zero
        s_rel = zero
        if self.use_rel:
            xt = self.trk_in(torch.cat([a_trk, m], -1))
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
        return dict(a_det=a_det, a_trk=a_trk, box_pred=box_pred, vel=vel, s_app=s_app, s_mot=s_mot, s_rel=s_rel, gates=g, s=s,
                    logP=self.assign(s, dv, tv), det_valid=dv, trk_valid=tv)

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
        new = mem.push(matched, box_obs, self.updater(mem.app, emb_obs), frame)
        if self.edges is not None:
            new = new.replace(edges=self.edges.update(mem.edges, box_obs, matched))
        return new, matched
