"""Extracción de queries del detector congelado (RT-DETRv2) y emparejamiento con el GT.

No modifica RT-DETRv2: un forward hook sobre la capa eval_idx del decoder captura su salida (B, num_queries, 256),
que es exactamente la entrada de las cabezas de score y caja, así que embedding, caja y score quedan alineados por query.
"""
import os, sys
import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'rtdetrv2_pytorch'))
from src.core import YAMLConfig
from src.zoo.rtdetr.box_ops import box_cxcywh_to_xyxy, box_iou

DEFAULT_CONFIG = os.path.join(ROOT, 'tracking', 'configs', 'detector_r50vd_person.yml')


def topk_queries(logits, boxes, emb, k):
    """Top-K queries por score (selección sin gradiente; el embedding conserva el suyo).
    logits (B,Q,C) boxes (B,Q,4) emb (B,Q,D) -> dict(emb, boxes, logits, scores, query_idx) con (B,K,...)."""
    scores = logits.detach().sigmoid().max(-1).values
    k = min(k, scores.shape[1])
    sc, idx = scores.topk(k, dim=1)
    g = lambda t: t.gather(1, idx.unsqueeze(-1).expand(-1, -1, t.shape[-1]))
    return dict(emb=g(emb), boxes=g(boxes), logits=g(logits), scores=sc, query_idx=idx)


@torch.no_grad()
def match_queries(matcher, q, gts, device, min_iou=0.0):
    """Asigna track ID a cada query con el Hungarian del detector (ver QueryExtractor.match)."""
    B, K = q['scores'].shape
    tid = torch.full((B, K), -1, dtype=torch.long)
    miou = torch.zeros(B, K)
    targets = [dict(labels=torch.zeros(len(g), dtype=torch.long, device=device), boxes=g[:, 1:5].to(device).float()) for g in gts]
    nonempty = [i for i, g in enumerate(gts) if len(g)]
    if not nonempty:
        return tid, miou
    sub = dict(pred_logits=q['logits'][nonempty].detach().float(), pred_boxes=q['boxes'][nonempty].detach().float())
    idx = matcher(sub, [targets[i] for i in nonempty])['indices']
    for b, (src, tgt) in zip(nonempty, idx):
        ious = box_iou(box_cxcywh_to_xyxy(q['boxes'][b, src].detach().float()), box_cxcywh_to_xyxy(targets[b]['boxes'][tgt]))[0].diag().cpu()
        keep = ious >= min_iou
        tid[b, src[keep]] = gts[b][tgt[keep], 0].long()
        miou[b, src[keep]] = ious[keep]
    return tid, miou


class QueryExtractor:
    """Detector congelado -> por frame, top-K queries: embedding (256), caja (cxcywh normalizada) y score."""

    def __init__(self, ckpt, config=DEFAULT_CONFIG, top_k=100, device='cuda:0'):
        cfg = YAMLConfig(config, PResNet=dict(pretrained=False))
        ck = torch.load(ckpt, map_location='cpu')
        model = cfg.model
        model.load_state_dict(ck['ema']['module'] if 'ema' in ck else ck['model'])
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.matcher = cfg.criterion.matcher  # HungarianMatcher del propio detector (mismos pesos de coste)
        self.top_k, self.device = top_k, device
        dec = self.model.decoder.decoder
        self._emb = None
        dec.layers[dec.eval_idx].register_forward_hook(lambda m, i, o: setattr(self, '_emb', o))

    @torch.no_grad()
    def __call__(self, images):
        """images: (B,3,640,640) en [0,1]. Devuelve dict con (B,K,...) : emb, boxes (cxcywh norm), scores, logits, query_idx."""
        out = self.model(images.to(self.device))
        logits, boxes, emb = out['pred_logits'], out['pred_boxes'], self._emb
        assert emb is not None and emb.shape[:2] == logits.shape[:2], 'el hook no capturó la salida del decoder'
        return topk_queries(logits, boxes, emb, self.top_k)

    def match(self, q, gts, min_iou=0.0):
        """Asigna track ID a cada query con el Hungarian del detector (la caché y la inferencia usan esto).
        q: salida de __call__; gts: lista (B) de tensores (N,5) = [track_id, cx, cy, w, h] normalizados.
        Devuelve track_id (B,K) con -1 = fondo, y match_iou (B,K). min_iou > 0 descarta emparejamientos con IoU menor."""
        return match_queries(self.matcher, q, gts, self.device, min_iou)
