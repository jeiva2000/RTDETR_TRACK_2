"""E2EModel (H1): tracker mínimo sobre q_det + caja. Estado de un track: apariencia a (se inicializa desde la query de su primera
detección), caja y último frame visto. Actualización de apariencia por EMA fija (appearance.alpha_app) solo con match.
Todo vectorizado por lote con padding + máscaras; sin gradiente a través del estado entre frames más allá del clip (BPTT del clip)."""
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tracking.assoc.assign import hungarian_assign
from tracking.e2e.heads import AffinityHead, BirthHead, box_iou_b, mlp
from tracking.e2e.state import TrackState


class E2EModel(nn.Module):
    def __init__(self, cfg, in_dim=256):
        super().__init__()
        D, H = cfg.heads.dim, cfg.heads.hidden
        self.cfg = cfg
        self.proj = nn.Sequential(nn.Linear(in_dim, D), nn.LayerNorm(D))
        self.enc = mlp(D, H, D)                    # lleva la apariencia de una detección al espacio del estado
        self.aff = AffinityHead(D, H, cfg.heads.tau_init)
        self.birth = BirthHead(D, H)
        self.alpha = float(cfg.appearance.alpha_app)
        self.birth_detach = bool(cfg.loss.get('birth_detach', True))

    def encode(self, emb):
        p = self.proj(emb)
        return p, self.enc(p)

    def init_state_vec(self, emb):
        """Apariencia inicial de un track nuevo = codificación de la query de su primera detección. emb (n,D_in) -> (n,D)."""
        return self.encode(emb)[1]

    def scores(self, state, dets, frame):
        """-> logits (B,N,K+1), birth_logit (B,K), enc (B,K,D). frame: (B,) frame actual de cada clip."""
        p, e = self.encode(dets.emb)
        logits = self.aff(state.a, state.box, state.gap(frame), p, dets.box, dets.score, dets.valid)
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
        """EMA de la apariencia y actualización de caja/último frame, solo para los tracks con match (assign < K)."""
        K = dets.emb.shape[1]
        matched = (assign < K) & state.alive
        j = assign.clamp(max=K - 1)
        enc_obs = torch.gather(enc, 1, j[..., None].expand(-1, -1, enc.shape[-1]))
        box_obs = torch.gather(dets.box, 1, j[..., None].expand(-1, -1, 4))
        a = torch.where(matched[..., None], (1 - self.alpha) * state.a + self.alpha * enc_obs, state.a)
        box = torch.where(matched[..., None], box_obs, state.box)
        last = torch.where(matched, frame[:, None].expand_as(state.last), state.last)
        return state.replace(a=a, box=box, last=last), matched
