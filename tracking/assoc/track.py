"""Tracker online con el módulo de asociación, sobre las features cacheadas de val. Escribe <out>/<seq>.txt (MOT).

Por frame: afinidad de cada track con las K queries + "no aparece" -> Hungarian global sobre log-probabilidades ->
actualización del estado. Nacimientos: queries no asignadas con score >= --birth-thresh que no solapan (IoU < --birth-nms)
con detecciones asignadas ni con otros nacimientos (NMS voraz por score). Muerte: más de --max-lost frames sin ver.
Variante hist: la visibilidad predicha puede ajustar cuánto se mantiene un track sin ver (--max-lost-occluded y --max-lost-out;
por defecto iguales a --max-lost, es decir, sin efecto).
Uso: python -m tracking.assoc.track --ckpt <assoc best.pth> --cache <val cache> --out <dir>; luego tracking.baseline.evaluate.
"""
import argparse, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tracking.assoc.model import AssocModel, box_iou_cxcywh
from tracking.assoc.assign import hungarian_assign
from tracking.assoc.core import Tracks, step
from tracking.baseline.mot_io import write_mot


@torch.no_grad()
def track_sequence(model, d, a, device):
    E_all, D_all, sc_all = d['emb'].to(device).float(), d['boxes'].to(device), d['scores'].to(device)
    W, H = d['meta']['width'], d['meta']['height']
    tr, next_id, rows = Tracks(model.proj[0].out_features, device, hist=model.hist_len if model.update_type == 'hist' else 0), 1, []
    for t in range(E_all.shape[0]):
        frame = t + 1
        E, D, sc = E_all[t], D_all[t], sc_all[t]
        used = torch.zeros(len(sc), dtype=torch.bool, device=device)
        if len(tr):
            _, idx = step(model, tr, E, D, sc, torch.tensor(frame, device=device), lambda lg: hungarian_assign(lg, a.min_prob))
            for i, j in enumerate(idx.tolist()):
                if j >= 0:
                    used[j] = True
                    cx, cy, w, h = D[j].tolist()
                    rows.append((frame, tr.ids[i], (cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H, float(sc[j])))
            if model.update_type == 'hist':  # tiempo de vida según la visibilidad predicha: 0 visible, 1 ocluido, 2 fuera
                v = tr.vis.argmax(-1)
                limit = torch.where(v == 1, torch.full_like(v, a.max_lost_occluded), torch.where(v == 2, torch.full_like(v, a.max_lost_out), torch.full_like(v, a.max_lost)))
            else:
                limit = a.max_lost
            tr.keep(frame - tr.last <= limit)
        cand = [j for j in torch.argsort(sc, descending=True).tolist() if not used[j] and sc[j] >= a.birth_thresh]
        taken = D[used]
        born = []
        for j in cand:
            ref = torch.cat([taken, D[born]]) if born else taken
            if len(ref) and box_iou_cxcywh(D[j:j + 1], ref).max() >= a.birth_nms:
                continue
            born.append(j)
        if born:
            j = torch.tensor(born, device=device)
            ids = list(range(next_id, next_id + len(born))); next_id += len(born)
            if model.update_type == 'hist':
                S0, p0 = model.birth(E[j])
                tr.add(ids, S0, D[j], frame, p=p0)
            else:
                tr.add(ids, model.init_state(E[j]), D[j], frame)
            for i, k in zip(ids, born):
                cx, cy, w, h = D[k].tolist()
                rows.append((frame, i, (cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H, float(sc[k])))
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True); p.add_argument('--cache', required=True); p.add_argument('--out', required=True)
    p.add_argument('--seqs', nargs='*', default=None)
    p.add_argument('--birth-thresh', type=float, default=0.5); p.add_argument('--birth-nms', type=float, default=0.7)
    p.add_argument('--max-lost-occluded', type=int, default=None, help='[hist] por defecto = --max-lost')
    p.add_argument('--max-lost-out', type=int, default=None, help='[hist] por defecto = --max-lost')
    p.add_argument('--max-lost', type=int, default=30); p.add_argument('--min-prob', type=float, default=0.0)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    ck = torch.load(a.ckpt, map_location='cpu'); ar = ck['args']
    a.max_lost_occluded = a.max_lost if a.max_lost_occluded is None else a.max_lost_occluded
    a.max_lost_out = a.max_lost if a.max_lost_out is None else a.max_lost_out
    model = AssocModel.from_args(ar, ck['model']['proj.0.weight'].shape[1]).to(a.device).eval()
    model.load_state_dict(ck['model'])
    seqs = a.seqs or sorted(f[:-3] for f in os.listdir(a.cache) if f.endswith('.pt'))
    for s in seqs:
        rows = track_sequence(model, torch.load(os.path.join(a.cache, f'{s}.pt')), a, a.device)
        write_mot(os.path.join(a.out, f'{s}.txt'), rows)
        print(f'{s}: {len(set(r[1] for r in rows))} tracks, {len(rows)} cajas', flush=True)
