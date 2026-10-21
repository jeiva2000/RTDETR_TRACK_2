"""Inferencia frame a frame del H3 sobre la caché de val: mantiene la memoria y exporta MOT (mismo formato y protocolo que tracking/e2e/infer.py).
Por frame: pasos 1-7 -> Hungarian sobre s con umbral de probabilidad (h3.min_prob) -> actualización de la memoria -> muerte por max_age -> nacimientos:
detecciones sin asignar con score >= h3.new_thr, con supresión de duplicados (IoU >= infer.birth_nms, voraz por score). Un track sin asignar se mantiene en
memoria hasta max_age (re-entrada) pero no se escribe mientras no se observa."""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from tracking.e2e.heads import box_iou_b
from tracking.e2e.tokens import from_cache


@torch.no_grad()
def track_sequence(model, d, cfg, device):
    from tracking.e2e.infer import _row
    emb, box, sc = d['emb'].to(device).float(), d['boxes'].to(device).float(), d['scores'].to(device).float()
    W, H = d['meta']['width'], d['meta']['height']
    mem = model.new_memory(1, device)
    next_id, rows, dropped = 1, [], 0
    for t in range(emb.shape[0]):
        frame = torch.tensor([t + 1], device=device)
        dets = from_cache(emb[t][None], box[t][None], sc[t][None])
        out = model.forward_scores(mem, dets, frame)
        assign = model.decide(out)
        mem, matched = model.update(mem, assign, dets, frame)
        used = torch.zeros(emb.shape[1], dtype=torch.bool, device=device)
        for i in matched[0].nonzero().flatten().tolist():
            j = int(assign[0, i]); used[j] = True
            rows.append(_row(t + 1, int(mem.ids[0, i]), box[t, j], sc[t, j], W, H))
        mem = mem.kill(frame, cfg.state.max_age)
        cand = (~used) & out['det_valid'][0] & (sc[t] >= cfg.h3.new_thr)
        taken = box[t][used]
        for j in torch.argsort(sc[t], descending=True).tolist():
            if not cand[j]:
                continue
            if len(taken) and box_iou_b(box[t, j][None, None], taken[None])[0, 0].max() >= cfg.infer.birth_nms:
                continue
            mem, made = mem.spawn(0, torch.tensor([next_id], device=device), model.init_app(emb[t, j][None]), box[t, j][None], frame[0])
            if made:
                rows.append(_row(t + 1, next_id, box[t, j], sc[t, j], W, H)); next_id += 1
                taken = torch.cat([taken, box[t, j][None]])
            else:
                dropped += 1
    return rows, dropped
