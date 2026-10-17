"""Datos sintéticos para validar el módulo de asociación SIN datos reales (no sustituyen a DanceTrack).

Genera, por secuencia, una caché como la de cache_features (emb, boxes, scores, track_id, match_iou) y un GT en formato MOT:
- P personas con una identidad (embedding) propia + ruido, que se mueven y se cruzan.
- Oclusiones LARGAS justo en los cruces: durante L frames una de las dos personas sigue en el GT pero el detector no la ve.
- Salidas de escena: una persona sale (deja de estar en el GT) durante unos frames y vuelve con el mismo ID.
- Queries de fondo con score bajo.
Uso: python -m tracking.tools.make_synthetic --out DIR --n-seq 8 --frames 300 --seed 0 [--occ-len 25]
Crea DIR/cache/<seq>.pt y DIR/mot/<seq>/{seqinfo.ini,gt/gt.txt}
"""
import argparse, os
import numpy as np
import torch


def make(out, name, N, P, K, occ_len, rng, W=1280, H=720):
    idv = rng.normal(size=(P, 256)) * 0.6
    x0 = rng.uniform(0.1, 0.9, P); vx = rng.uniform(-0.006, 0.006, P); y = rng.uniform(0.3, 0.7, P)
    pos = lambda i, f: ((x0[i] + vx[i] * f) % 1.0, y[i] + 0.02 * np.sin(f / 10 + i))
    absent = np.zeros((P, N), bool)   # el detector no la ve (sigue en el GT)
    gone = np.zeros((P, N), bool)     # fuera de escena (no está en el GT)
    for f in range(1, N):
        for i in range(P):
            for j in range(i + 1, P):
                (xi, yi), (xj, yj) = pos(i, f), pos(j, f)
                if abs(xi - xj) < 0.04 and abs(yi - yj) < 0.05 and rng.random() < 0.3:   # cruce: oclusión larga de una de las dos
                    v = i if rng.random() < 0.5 else j
                    absent[v, f:f + rng.integers(occ_len // 2, occ_len + 1)] = True
    for i in range(P):
        if rng.random() < 0.5:
            a = rng.integers(20, max(21, N - 40)); gone[i, a:a + rng.integers(10, 30)] = True
    emb = np.zeros((N, K, 256), np.float32); box = np.zeros((N, K, 4), np.float32); sc = np.zeros((N, K), np.float32)
    tid = -np.ones((N, K), np.int32); miou = np.zeros((N, K), np.float32); gt = []
    for f in range(N):
        perm = rng.permutation(K); k = 0
        for i in range(P):
            if gone[i, f]:
                continue
            cx, cy = pos(i, f)
            gt.append([f + 1, i, (cx - 0.04) * W, (cy - 0.15) * H, 0.08 * W, 0.3 * H, 1, 1, 1])
            if absent[i, f] or rng.random() < 0.03:
                continue
            j = perm[k]; k += 1
            emb[f, j] = idv[i] + rng.normal(size=256) * 0.5; box[f, j] = [cx, cy, 0.08, 0.3] + rng.normal(size=4) * 0.003
            sc[f, j] = rng.uniform(0.6, 0.95); tid[f, j] = i; miou[f, j] = 0.9
        for j in perm[k:]:
            emb[f, j] = rng.normal(size=256) * 0.6; box[f, j] = [rng.uniform(0, 1), rng.uniform(0.2, 0.8), 0.08, 0.3]
            sc[f, j] = rng.uniform(0.0, 0.3)
    os.makedirs(f'{out}/cache', exist_ok=True)
    torch.save(dict(emb=torch.tensor(emb).half(), boxes=torch.tensor(box), scores=torch.tensor(sc), track_id=torch.tensor(tid),
                    match_iou=torch.tensor(miou), meta=dict(top_k=K, width=W, height=H)), f'{out}/cache/{name}.pt')
    d = f'{out}/mot/{name}'; os.makedirs(d + '/gt', exist_ok=True)
    open(d + '/seqinfo.ini', 'w').write(f'[Sequence]\nname={name}\nimDir=img1\nframeRate=20\nseqLength={N}\nimWidth={W}\nimHeight={H}\nimExt=.jpg\n')
    np.savetxt(d + '/gt/gt.txt', np.array(gt), delimiter=',', fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')
    return int(absent.sum()), int(gone.sum())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True); p.add_argument('--n-seq', type=int, default=8); p.add_argument('--frames', type=int, default=300)
    p.add_argument('--persons', type=int, default=6); p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--occ-len', type=int, default=25); p.add_argument('--seed', type=int, default=0); p.add_argument('--prefix', default='syn')
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    for s in range(a.n_seq):
        o, g = make(a.out, f'{a.prefix}{s:02d}', a.frames, a.persons, a.top_k, a.occ_len, rng)
        print(f'{a.prefix}{s:02d}: persona-frames ocluidas {o}, fuera de escena {g}')
