"""¿Qué pasa en los cambios de identidad? Clasifica cada IDSW de uno o varios resultados de tracking (formato MOT) frente al GT de val.

Emparejamiento por frame como CLEAR-MOT (TrackEval): Hungarian sobre IoU >= --iou, con preferencia por mantener el emparejamiento anterior
de cada persona GT. Un cambio de identidad (IDSW) es que la persona GT g, emparejada por última vez con la predicción q en el frame f0,
quede emparejada con otra p != q en el frame f. El total coincide aproximadamente con el IDSW de TrackEval. Por cada IDSW:
- hueco = f − f0 − 1: frames sin emparejar entre medias (persona ocluida, no detectada, o el tracker no escribió su caja).
- tipo:
  * renacimiento: p es un id NUEVO, cuya primera caja es posterior a f0. El track q se perdió y la persona volvió con otro id.
    Por tramo de hueco: 0 (sin hueco: se rompió con la persona a la vista), 1-10 (dentro de lo entrenado, k <= 10), 11-30, >30
    (más allá de max_lost=30: q ya había muerto, solo una memoria más larga lo arregla).
  * robo: p ya existía antes de f0, es decir, era otro track que pasa a seguir a g. Si en ese mismo frame q está emparejado con otra
    persona GT, es un intercambio (swap) entre dos tracks vivos. Si no, p dejó a su persona o era un falso positivo.
- q_vive: q vuelve a escribir alguna caja después de f (seguía vivo); si no, q terminó.
- cruce: la caja GT de g solapa (IoU > --cross-iou) con otra persona en algún frame de [f0, f].
- vuelve: más adelante g vuelve a emparejarse con q (cambio temporal).
Uso: python -m tracking.trackheads.idsw --gt-root $DANCETRACK_DIR/val nombre=<dir con <seq>.txt> [nombre2=<dir> ...] [--out idsw.json]"""
import argparse, json, os, sys
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tracking.baseline.mot_io import by_frame, read_mot

GAPS = [(0, 0, '0'), (1, 10, '1-10'), (11, 30, '11-30'), (31, 10 ** 9, '>30')]


def iou_xywh(a, b):
    """a (n,4), b (m,4) en x,y,w,h -> (n,m)."""
    ax1, ay1, ax2, ay2 = a[:, 0, None], a[:, 1, None], (a[:, 0] + a[:, 2])[:, None], (a[:, 1] + a[:, 3])[:, None]
    bx1, by1, bx2, by2 = b[None, :, 0], b[None, :, 1], b[None, :, 0] + b[None, :, 2], b[None, :, 1] + b[None, :, 3]
    inter = (np.minimum(ax2, bx2) - np.maximum(ax1, bx1)).clip(0) * (np.minimum(ay2, by2) - np.maximum(ay1, by1)).clip(0)
    return inter / (a[:, 2, None] * a[:, 3, None] + b[None, :, 2] * b[None, :, 3] - inter + 1e-9)


def match_sequence(gt, pr, thr=0.5):
    """gt, pr: arrays MOT. -> {frame: {gt_id: pred_id}} con el criterio de CLEAR (continuidad primero)."""
    G, P = by_frame(gt), by_frame(pr)
    prev, out = {}, {}
    for f in sorted(G):
        g, p = G[f], P.get(f)
        if p is None or not len(p):
            out[f] = {}
            continue
        sim = iou_xywh(g[:, 2:6], p[:, 2:6])
        gid, pid = g[:, 1].astype(int), p[:, 1].astype(int)
        cont = np.array([[prev.get(a) == b for b in pid] for a in gid])
        score = np.where(sim >= thr - 1e-9, sim + 1000 * cont, 0)
        r, c = linear_sum_assignment(-score)
        m = {int(gid[i]): int(pid[j]) for i, j in zip(r, c) if score[i, j] > 0}
        out[f] = m
        prev.update(m)
    return out


def sequence_events(gt, pr, thr=0.5, cross_iou=0.3):
    M = match_sequence(gt, pr, thr)
    first = {int(i): int(pr[pr[:, 1] == i, 0].min()) for i in np.unique(pr[:, 1])} if len(pr) else {}
    frames_of = defaultdict(set)
    for r in pr:
        frames_of[int(r[1])].add(int(r[0]))
    G = by_frame(gt)
    # solape máximo de cada persona GT con otra en cada frame
    over = {}
    for f, g in G.items():
        s = iou_xywh(g[:, 2:6], g[:, 2:6])
        np.fill_diagonal(s, 0)
        over[f] = dict(zip(g[:, 1].astype(int), s.max(1) if len(g) > 1 else np.zeros(len(g))))
    ev, last = [], {}
    hist = defaultdict(list)                                           # gt -> [(frame, pred)]
    for f in sorted(M):
        inv = {p: g for g, p in M[f].items()}
        for g, p in M[f].items():
            if g in last and last[g][1] != p:
                f0, q = last[g]
                ev.append(dict(gt=g, frame=f, f0=f0, q=q, p=p, gap=f - f0 - 1, new=first[p] > f0,
                               swap=q in inv and inv[q] != g, q_alive=any(x > f for x in frames_of[q]),
                               cross=any(over.get(x, {}).get(g, 0) > cross_iou for x in range(f0, f + 1))))
            last[g] = (f, p)
            hist[g].append((f, p))
    for e in ev:                                                       # ¿vuelve más tarde a q?
        e['back'] = any(fr > e['frame'] and pp == e['q'] for fr, pp in hist[e['gt']])
    return ev


def summarize(ev):
    n = len(ev)
    if not n:
        return dict(idsw=0)
    pct = lambda k: round(100 * k / n, 1)
    s = dict(idsw=n)
    ren = [e for e in ev if e['new']]
    rob = [e for e in ev if not e['new']]
    s['renacimiento_%'] = pct(len(ren))
    for lo, hi, name in GAPS:
        s[f'  hueco {name}'] = pct(sum(lo <= e['gap'] <= hi for e in ren))
    s['  q sigue vivo'] = pct(sum(e['q_alive'] for e in ren))
    s['robo_%'] = pct(len(rob))
    s['  intercambio (swap)'] = pct(sum(e['swap'] for e in rob))
    for lo, hi, name in GAPS:
        s[f'  hueco {name} '] = pct(sum(lo <= e['gap'] <= hi for e in rob))
    s['en cruce_%'] = pct(sum(e['cross'] for e in ev))
    s['vuelve a q_%'] = pct(sum(e['back'] for e in ev))
    s['hueco mediana'] = float(np.median([e['gap'] for e in ev]))
    return s


def main(a):
    seqs = sorted(d for d in os.listdir(a.gt_root) if os.path.isfile(os.path.join(a.gt_root, d, 'gt', 'gt.txt')))
    res, allev = {}, {}
    for item in a.results:
        name, d = item.split('=', 1)
        ev = []
        for s in seqs:
            pf = os.path.join(d, f'{s}.txt')
            if not os.path.isfile(pf):
                continue
            gt = read_mot(os.path.join(a.gt_root, s, 'gt', 'gt.txt'))
            gt = gt[gt[:, 6] != 0]
            for e in sequence_events(gt, read_mot(pf), a.iou, a.cross_iou):
                e['seq'] = s
                ev.append(e)
        res[name], allev[name] = summarize(ev), ev
    names = list(res)
    keys = list(dict.fromkeys(k for r in res.values() for k in r))
    print('| | ' + ' | '.join(names) + ' |\n|---' * 1 + '|---' * len(names) + '|')
    for k in keys:
        print(f'| {k} | ' + ' | '.join(str(res[n].get(k, '')) for n in names) + ' |')
    print('\n(% sobre el total de IDSW de cada método; los subapartados de renacimiento y robo son % del total, no del grupo)')
    if a.out:
        json.dump(dict(summary=res, events={n: [{k: (int(v) if isinstance(v, (bool, np.bool_, np.integer)) else v) for k, v in e.items()}
                                                 for e in ev] for n, ev in allev.items()}), open(a.out, 'w'), indent=0)
        print('guardado en', a.out)
    return res, allev


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('results', nargs='+', help='nombre=<carpeta con <seq>.txt>')
    p.add_argument('--gt-root', default=os.environ.get('DANCETRACK_DIR', '') and os.path.join(os.environ.get('DANCETRACK_DIR', ''), 'val'))
    p.add_argument('--iou', type=float, default=0.5)
    p.add_argument('--cross-iou', type=float, default=0.3)
    p.add_argument('--out', default=None, help='JSON con el resumen y la lista de eventos (seq, gt, frame, f0, q, p, gap, ...)')
    return p


if __name__ == '__main__':
    main(parser().parse_args())
