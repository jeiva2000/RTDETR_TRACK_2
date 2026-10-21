"""E2EModel (H1+H2): tracker sobre q_det + caja. Estado de un track: apariencia a (se inicializa desde la query de su primera
detección), caja y último frame visto. Actualización de apariencia por EMA fija (appearance.alpha_app) solo con match.
H2 (motion.mode=gru): segundo vector por track, m (GRU sobre los deltas de caja observados), del que sale una predicción de movimiento con σ
(media y log σ del delta de caja para el frame actual); entra en la afinidad (z, NLL, IoU con la caja predicha) y en la columna nula.
Con motion.mode=none el modelo es exactamente el H1 (mismos parámetros).
Todo vectorizado por lote con padding + máscaras; sin gradiente a través del estado entre frames más allá del clip (BPTT del clip)."""
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tracking.assoc.assign import hungarian_assign
from tracking.e2e.heads import AffinityHead, BirthHead, box_iou_b, mlp
from tracking.e2e.motion import EXTRA_DIM, MotionHead, motion_features
from tracking.e2e.state import TrackState


class E2EModel(nn.Module):
    def __init__(self, cfg, in_dim=256):
        super().__init__()
        D, H = cfg.heads.dim, cfg.heads.hidden
        self.cfg = cfg
        self.proj = nn.Sequential(nn.Linear(in_dim, D), nn.LayerNorm(D))
        self.enc = mlp(D, H, D)                    # lleva la apariencia de una detección al espacio del estado
        mc = cfg.motion
        assert mc.mode in ('none', 'gru'), f'motion.mode desconocido: {mc.mode}'
        self.motion = MotionHead(mc.dim, mc.hidden, mc.vel_scale) if mc.mode == 'gru' else None
        self.aff = AffinityHead(D, H, cfg.heads.tau_init, EXTRA_DIM if self.motion else 0, (mc.dim + 1) if self.motion else 0)
        self.birth = BirthHead(D, H)
        self.alpha = float(cfg.appearance.alpha_app)
        self.birth_detach = bool(cfg.loss.get('birth_detach', True))

    def new_state(self, B, device):
        return TrackState.empty(B, self.cfg.state.max_tracks, self.cfg.heads.dim, device, self.motion.dim if self.motion else 0)

    def predict_motion(self, state, frame):
        """(mu, log σ) del delta de caja de cada track para el frame actual, o None sin módulo de movimiento."""
        return None if self.motion is None else self.motion.predict(state.m, state.gap(frame))

    def encode(self, emb):
        p = self.proj(emb)
        return p, self.enc(p)

    def init_state_vec(self, emb):
        """Apariencia inicial de un track nuevo = codificación de la query de su primera detección. emb (n,D_in) -> (n,D)."""
        return self.encode(emb)[1]

    def scores(self, state, dets, frame, pred=None):
        """-> logits (B,N,K+1), birth_logit (B,K), enc (B,K,D). frame: (B,) frame actual de cada clip.
        pred: (mu, log σ) ya calculado con predict_motion (el entrenamiento lo reutiliza para la pérdida); si es None se calcula aquí."""
        p, e = self.encode(dets.emb)
        gap, extra, null_extra = state.gap(frame), None, None
        if self.motion is not None:
            mu, logsig = pred if pred is not None else self.motion.predict(state.m, gap)
            extra = motion_features(state.box, mu, logsig, dets.box)
            null_extra = torch.cat([state.m, logsig.mean(-1, keepdim=True)], -1)
        logits = self.aff(state.a, state.box, gap, p, dets.box, dets.score, dets.valid, extra, null_extra)
        with torch.no_grad():
            prob = torch.softmax(logits, -1)[..., :-1] * state.alive[..., None]
            claim = prob.max(1).values                                       # (B,K) lo que reclama el mejor track
            iou_max = (box_iou_b(state.box, dets.box) * state.alive[..., None]).max(1).values
        p_b = p.detach() if self.birth_detach else p     # por defecto el nacimiento NO moldea la apariencia compartida
        return logits, self.birth(p_b, dets.score, iou_max, claim), e

    @staticmethod
    def decide(logits, alive, target=None, use_model=None):
        """Asignación (B,N) con valores en [0, K]; K = columna nula. Con target (teacher forcing) se usa donde use_model es False;
        donde es True, Hungarian global sobre log-probabilidades (cada detección a un solo track; cada track puede elegir 'no aparece')."""
        B, N, K1 = logits.shape
        K = K1 - 1
        out = target.clone() if target is not None else torch.full((B, N), K, dtype=torch.long, device=logits.device)
        if use_model is None:
            use_model = torch.ones(B, dtype=torch.bool)
        for b in range(B):
            if not bool(use_model[b]):
                continue
            rows = alive[b].nonzero().flatten()
            if len(rows) == 0:
                continue
            idx, _ = hungarian_assign(logits[b, rows])
            out[b, rows] = torch.where(idx >= 0, idx, torch.full_like(idx, K)).to(out.device)
        return out

    def update(self, state, assign, dets, enc, frame):
        """EMA de la apariencia, GRU de movimiento (H2) y actualización de caja/último frame, solo para los tracks con match (assign < K)."""
        K = dets.emb.shape[1]
        matched = (assign < K) & state.alive
        j = assign.clamp(max=K - 1)
        enc_obs = torch.gather(enc, 1, j[..., None].expand(-1, -1, enc.shape[-1]))
        box_obs = torch.gather(dets.box, 1, j[..., None].expand(-1, -1, 4))
        a = torch.where(matched[..., None], (1 - self.alpha) * state.a + self.alpha * enc_obs, state.a)
        box = torch.where(matched[..., None], box_obs, state.box)
        last = torch.where(matched, frame[:, None].expand_as(state.last), state.last)
        m = state.m if self.motion is None else self.motion.observe(state.m, state.box, box_obs, frame[:, None] - state.last, matched)
        return state.replace(a=a, box=box, last=last, m=m), matched


def build_model(cfg, in_dim=256):
    """E2EModel (H1/H2) o H3Model (h3.enabled=true): el resto del pipeline (entrenamiento, inferencia, barrido) solo usa esta fábrica."""
    if cfg.h3.enabled:
        from tracking.e2e.h3.model import H3Model
        return H3Model(cfg, in_dim)
    return E2EModel(cfg, in_dim)
