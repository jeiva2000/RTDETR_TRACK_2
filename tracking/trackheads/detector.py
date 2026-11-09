"""Detector congelado para TrackHeads: lo mismo que QueryExtractor (top-K queries de la capa eval_idx del decoder, emparejadas con el GT por
el Hungarian del detector) y, además, las 3 salidas del encoder híbrido (B,256,H,W), strides 8/16/32, capturadas con un forward hook.
No modifica RT-DETRv2."""
import torch

from tracking.video.extractor import DEFAULT_CONFIG, QueryExtractor


class FrozenDetector(QueryExtractor):
    def __init__(self, ckpt, config=DEFAULT_CONFIG, top_k=100, device='cuda:0', amp=False):
        super().__init__(ckpt, config, top_k, device)
        self.amp = amp
        self._enc = None
        self.model.encoder.register_forward_hook(lambda m, i, o: setattr(self, '_enc', o))

    @torch.no_grad()
    def detect(self, images, gts, min_iou=0.5):
        """images (B,3,640,640) en [0,1]; gts lista (B) de (n,5) [id, cx, cy, w, h].
        -> q (dict de QueryExtractor: emb, boxes, scores, ... (B,K,...)), track_id (B,K) con -1 = fondo, feats [3×(B,256,H,W)] float32."""
        with torch.autocast('cuda', enabled=self.amp and str(self.device).startswith('cuda')):
            q = self(images)
        feats = [f.float() for f in self._enc]
        assert len(feats) == 3, f'el hook del encoder capturó {len(feats)} niveles'
        q = {k: v.float() if v.is_floating_point() else v for k, v in q.items()}
        tid, _ = self.match(q, gts, min_iou)
        return q, tid.to(q['scores'].device), feats
