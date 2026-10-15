"""Subconjunto de cruces y estadísticas de "una predicción cubre a dos personas GT".

Cruce: frame donde algún par de cajas GT tiene IoU > iou_thr; se añade un margen de ±margin frames.
Los frames resultantes se agrupan en segmentos contiguos; cada segmento se evalúa como una pseudo-secuencia
independiente (frames renumerados desde 1), así que IDSW/AssA se miden dentro de cada ventana de cruce.
"""
import json, os
import numpy as np
from .mot_io import read_mot, by_frame, write_mot


def iou_matrix(a, b):
    """a: (N,4) b: (M,4) en x,y,w,h -> (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(ax1[:, None], bx1[None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(ay1[:, None], by1[None]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:3] * a[:, 3:4] + (b[:, 2] * b[:, 3])[None] - inter + 1e-9)


def crossing_frames(gt_frames, n_frames, iou_thr=0.3, margin=10):
    """gt_frames: {frame: filas MOT}. Devuelve array ordenado de frames (1-indexados) del subconjunto."""
    hit = np.zeros(n_frames + 2, bool)
    for fr, rows in gt_frames.items():
        if len(rows) < 2 or fr > n_frames:
            continue
        m = iou_matrix(rows[:, 2:6], rows[:, 2:6]); np.fill_diagonal(m, 0)
        if (m > iou_thr).any():
            hit[fr] = True
    out = np.zeros_like(hit)
    for fr in np.where(hit)[0]:
        out[max(1, fr - margin): min(n_frames, fr + margin) + 1] = True
    return np.where(out)[0]


def segments(frames):
    """frames ordenados -> lista de (inicio, fin) de tramos contiguos"""
    segs = []
    for f in frames:
        if segs and f == segs[-1][1] + 1:
            segs[-1][1] = int(f)
        else:
            segs.append([int(f), int(f)])
    return [tuple(s) for s in segs]


def build_subset(gt_root, seqs, out_root, iou_thr=0.3, margin=10, seq_lengths=None):
    """Escribe <out_root>/<seq>_s<inicio>/gt/gt.txt (frames renumerados) e index.json. Devuelve el índice."""
    index = {}
    for s in seqs:
        gt = read_mot(os.path.join(gt_root, s, 'gt', 'gt.txt'), full=True)
        gtf = by_frame(gt)
        n = seq_lengths[s]
        for (a, b) in segments(crossing_frames(gtf, n, iou_thr, margin)):
            name = f'{s}_s{a:05d}'
            sel = gt[(gt[:, 0] >= a) & (gt[:, 0] <= b)].copy(); sel[:, 0] -= a - 1
            os.makedirs(os.path.join(out_root, name, 'gt'), exist_ok=True)
            np.savetxt(os.path.join(out_root, name, 'gt', 'gt.txt'), sel, delimiter=',',
                       fmt=['%d', '%d'] + ['%.2f'] * 4 + ['%g'] * (sel.shape[1] - 6))
            index[name] = dict(seq=s, start=a, end=b, length=b - a + 1)
    json.dump(index, open(os.path.join(out_root, 'index.json'), 'w'), indent=1)
    return index


def cut_results(results_dir, index, out_dir):
    """Recorta los resultados del tracker a los segmentos del índice (misma renumeración que el GT)."""
    os.makedirs(out_dir, exist_ok=True)
    for name, e in index.items():
        r = read_mot(os.path.join(results_dir, f'{e["seq"]}.txt'))
        r = r[(r[:, 0] >= e['start']) & (r[:, 0] <= e['end'])]
        write_mot(os.path.join(out_dir, f'{name}.txt'), [(x[0] - e['start'] + 1, x[1], x[2], x[3], x[4], x[5], x[6]) for x in r])


def merged_stats(gt_root, pred_dir, index, iou_thrs=(0.3, 0.5), min_score=0.0):
    """En los frames de cruce: cuántas predicciones tienen IoU >= thr con >= 2 personas GT distintas.
    Devuelve {thr: {'merged': n, 'preds': n_predicciones, 'frames': n_frames_con_alguna}} y totales."""
    res = {t: dict(merged=0, preds=0, frames=0) for t in iou_thrs}
    n_frames = 0
    seqs = {}
    for e in index.values():
        seqs.setdefault(e['seq'], []).append((e['start'], e['end']))
    for s, segs in seqs.items():
        gtf = by_frame(read_mot(os.path.join(gt_root, s, 'gt', 'gt.txt')))
        prf = by_frame(read_mot(os.path.join(pred_dir, f'{s}.txt')))
        for (a, b) in segs:
            for fr in range(a, b + 1):
                n_frames += 1
                g = gtf.get(fr); p = prf.get(fr)
                if p is None or g is None or len(p) == 0 or len(g) == 0:
                    continue
                p = p[p[:, 6] >= min_score]
                m = iou_matrix(p[:, 2:6], g[:, 2:6])
                for t in iou_thrs:
                    k = int(((m >= t).sum(1) >= 2).sum())
                    res[t]['merged'] += k; res[t]['preds'] += len(p); res[t]['frames'] += int(k > 0)
    return dict(crossing_frames=n_frames, by_iou={str(t): v for t, v in res.items()})
