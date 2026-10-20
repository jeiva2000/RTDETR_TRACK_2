"""Inferencia frame a frame del módulo e2e (H1/H2) sobre la caché de val: mantiene el estado y exporta MOT (<out>/<seq>.txt) para evaluar
con tracking/baseline/evaluate.py (mismo protocolo que ByteTrack/OC-SORT y las variantes de assoc/).
Por frame: afinidad -> Hungarian global -> actualización -> muerte por max_age -> nacimientos: detecciones sin asignar con probabilidad de la
cabeza de nacimiento >= infer.birth_thr (infer.birth_mode=head) y score >= infer.min_score; con infer.birth_mode=score solo cuenta la regla de score, con supresión de duplicados (IoU >= infer.birth_nms, voraz por score).
Uso: python -m tracking.e2e.infer --ckpt <best.pth> --cache <val cache> --out <dir> [-u infer.birth_thr=0.6]"""
import argparse, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tracking.baseline.mot_io import write_mot
from tracking.e2e.config import load_config
from tracking.e2e.heads import box_iou_b
from tracking.e2e.model import E2EModel
from tracking.e2e.tokens import from_cache


def _row(frame, tid, box, score, W, H):
    cx, cy, w, h = box.tolist()
    return (frame, tid, (cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H, float(score))


@torch.no_grad()
def track_sequence(model, d, cfg, device):
    emb, box, sc = d['emb'].to(device).float(), d['boxes'].to(device).float(), d['scores'].to(device).float()
    W, H = d['meta']['width'], d['meta']['height']
    state = model.new_state(1, device)
    next_id, rows, dropped = 1, [], 0
    for t in range(emb.shape[0]):
        frame = torch.tensor([t + 1], device=device)
        dets = from_cache(emb[t][None], box[t][None], sc[t][None])
        logits, birth_logit, enc = model.scores(state, dets, frame)
        assign = model.decide(logits, state.alive)
        state, matched = model.update(state, assign, dets, enc, frame)
        used = torch.zeros(emb.shape[1], dtype=torch.bool, device=device)
        for i in matched[0].nonzero().flatten().tolist():
            j = int(assign[0, i]); used[j] = True
            rows.append(_row(t + 1, int(state.ids[0, i]), box[t, j], sc[t, j], W, H))
        state = state.kill(frame, cfg.state.max_age)
        cand = (~used) & (sc[t] >= cfg.infer.min_score)
        if cfg.infer.birth_mode == 'head':   # 'score': solo la regla (sin asignar + score + NMS), como tracking/assoc/track.py
            cand &= torch.sigmoid(birth_logit[0]) >= cfg.infer.birth_thr
        taken = box[t][used]
        for j in torch.argsort(sc[t], descending=True).tolist():
            if not cand[j]:
                continue
            if len(taken) and box_iou_b(box[t, j][None, None], taken[None])[0, 0].max() >= cfg.infer.birth_nms:
                continue
            ids = torch.tensor([next_id], device=device)
            state, made = state.spawn(0, ids, model.init_state_vec(emb[t, j][None]), box[t, j][None], frame[0])
            if made:
                rows.append(_row(t + 1, next_id, box[t, j], sc[t, j], W, H)); next_id += 1
                taken = torch.cat([taken, box[t, j][None]])
            else:
                dropped += 1
    return rows, dropped


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True); p.add_argument('--cache', required=True); p.add_argument('--out', required=True)
    p.add_argument('--seqs', nargs='*', default=None)
    p.add_argument('-u', '--update', nargs='*', default=[], help='sobrescribe claves: infer.birth_thr=0.6 state.max_age=30')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    ck = torch.load(a.ckpt, map_location='cpu')
    cfg = load_config(None, ck['cfg'])                  # la configuración del checkpoint ...
    if a.update:                                        # ... con los -u de la línea de comandos por encima
        from tracking.e2e.config import deep_update, parse_overrides
        deep_update(cfg, parse_overrides(a.update))
    model = E2EModel(cfg, ck['in_dim']).to(a.device).eval()
    model.load_state_dict(ck['model'])
    seqs = a.seqs or sorted(f[:-3] for f in os.listdir(a.cache) if f.endswith('.pt'))
    for s in seqs:
        rows, dropped = track_sequence(model, torch.load(os.path.join(a.cache, f'{s}.pt')), cfg, a.device)
        write_mot(os.path.join(a.out, f'{s}.txt'), rows)
        print(f'{s}: {len(set(r[1] for r in rows))} tracks, {len(rows)} cajas' + (f', {dropped} nacimientos perdidos (sin hueco)' if dropped else ''), flush=True)
