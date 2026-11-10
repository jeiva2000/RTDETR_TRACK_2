"""Tracker en línea con TrackHeads sobre secuencias completas (DanceTrack val), para evaluar con TrackEval (HOTA, AssA, IDF1...)
con el mismo protocolo que el baseline (tracking/baseline/evaluate.py: global y subconjunto de cruces).

Un forward del detector congelado por frame (en lotes) da las queries (emb, caja, score) y los mapas del encoder. Varios trackers comparten
esas detecciones y solo cambian en la puntuación de asociación (--modes):
- head: TrackHeads + Associator. Cada track vivo genera su cabeza a partir de (emb, caja) de su última detección asignada y puntúa las
  candidatas del frame actual (score >= --det-thr) y "ninguna"; puntuación = log-softmax de assoc_logits.
- iou: la misma maquinaria con puntuación = IoU(caja de la última detección, candidata) y "ninguna" = --iou-min. Es la referencia que
  aísla lo que aporta la cabeza (mismo detector, nacimientos y muertes).
Asignación: Hungarian global (cada candidata a un track como mucho; cada track puede quedarse en "ninguna" con su coste propio).
Tracks asignados: emb y caja pasan a ser los de la detección asignada (como en el entrenamiento, el track es la detección de su último frame).
Nacimientos: candidatas sin asignar con score >= --birth-thresh que no solapan (IoU < --birth-nms) con las asignadas ni con otros
nacimientos (NMS voraz por score). Muerte: más de --max-lost frames sin asignación. Solo se escriben las cajas asignadas o nacidas en cada frame.
Uso: python -m tracking.trackheads.track --ckpt <trackheads best.pth> --out <dir> [--modes head iou]; luego tracking.baseline.evaluate
sobre <dir>/<modo>/. O todo junto con slurm/trackheads_track.sbatch."""
import argparse, os, sys, time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tracking.baseline.mot_io import read_seqinfo, write_mot
from tracking.trackheads.data import seq_dirs
from tracking.trackheads.model import pair_iou
from tracking.trackheads.train import build_model
from tracking.video.frames import SequenceFrames

BIG = 1e9


def assign(score, null):
    """score (T,C) a maximizar, null (T,) puntuación de quedarse sin asignar -> idx (T,) long con la candidata o -1."""
    T, C = score.shape
    idx = torch.full((T,), -1, dtype=torch.long)
    if T == 0 or C == 0:
        return idx
    cost = np.full((T, C + T), BIG)
    cost[:, :C] = -score.double().cpu().numpy()
    cost[np.arange(T), C + np.arange(T)] = -null.double().cpu().numpy()
    r, c = linear_sum_assignment(cost)
    for i, j in zip(r, c):
        if j < C:
            idx[i] = int(j)
    return idx


class Tracker:
    """Estado de un tracker: por track, id, emb y caja de su última detección asignada y el frame en que se asignó."""

    def __init__(self, mode, model, a, dim=256, device='cpu'):
        self.mode, self.model, self.a, self.dev = mode, model, a, device
        self.ids, self.emb, self.box, self.last = [], torch.zeros(0, dim, device=device), torch.zeros(0, 4, device=device), torch.zeros(0, device=device)
        self.next_id = 1

    def scores(self, emb_c, box_c, feats):
        """-> score (T,C) y null (T,) para los tracks vivos frente a las candidatas."""
        if self.mode == 'iou':
            return pair_iou(self.box[None], box_c[None])[0], torch.full((len(self.ids),), self.a.iou_min, device=self.dev)
        T = len(self.ids)
        mask = torch.ones(1, T, dtype=torch.bool, device=self.dev)
        out = self.model(self.emb[None], self.box[None], [f[None] for f in feats], mask)
        self.model.associate(out, emb_c[None], box_c[None], torch.ones(1, len(box_c), dtype=torch.bool, device=self.dev))
        logp = out['assoc_logits'][0].float().log_softmax(-1)
        return logp[:, :-1], logp[:, -1] + self.a.null_bias

    def step(self, frame, emb_c, box_c, sc_c, feats):
        """Un frame. emb_c (C,D), box_c (C,4), sc_c (C,) candidatas -> lista de (id, j) asignadas o nacidas."""
        a, out = self.a, []
        used = torch.zeros(len(sc_c), dtype=torch.bool, device=self.dev)
        if self.ids:
            score, null = self.scores(emb_c, box_c, feats)
            idx = assign(score, null)
            for i, j in enumerate(idx.tolist()):
                if j >= 0:
                    used[j] = True
                    self.emb[i], self.box[i], self.last[i] = emb_c[j], box_c[j], frame
                    out.append((self.ids[i], j))
            keep = (frame - self.last) <= a.max_lost
            self.ids = [t for t, k in zip(self.ids, keep.tolist()) if k]
            self.emb, self.box, self.last = self.emb[keep], self.box[keep], self.last[keep]
        born = []
        for j in torch.argsort(sc_c, descending=True).tolist():
            if used[j] or sc_c[j] < a.birth_thresh:
                continue
            ref = torch.cat([box_c[used], box_c[born]]) if born else box_c[used]
            if len(ref) and pair_iou(box_c[j:j + 1][None], ref[None]).max() >= a.birth_nms:
                continue
            born.append(j)
        if born:
            j = torch.tensor(born, device=self.dev)
            new = list(range(self.next_id, self.next_id + len(born)))
            self.next_id += len(born)
            self.ids += new
            self.emb, self.box = torch.cat([self.emb, emb_c[j]]), torch.cat([self.box, box_c[j]])
            self.last = torch.cat([self.last, torch.full((len(born),), float(frame), device=self.dev)])
            out += list(zip(new, born))
        return out


@torch.no_grad()
def track_sequence(det, model, seq_dir, a, device):
    """-> {modo: filas MOT (frame, id, x, y, w, h, score)} de una secuencia."""
    info = read_seqinfo(seq_dir)
    W, H = info['width'], info['height']
    trackers = {m: Tracker(m, model, a, device=device) for m in a.modes}
    rows = {m: [] for m in a.modes}
    dl = torch.utils.data.DataLoader(SequenceFrames(seq_dir), batch_size=a.batch, num_workers=a.num_workers, shuffle=False)
    for imgs, frames in dl:
        q, feats = det.infer(imgs.to(device))
        for b, frame in enumerate(frames.tolist()):
            cand = q['scores'][b] >= a.det_thr
            emb_c, box_c, sc_c = q['emb'][b][cand], q['boxes'][b][cand], q['scores'][b][cand]
            fb = [f[b] for f in feats]
            for m, tr in trackers.items():
                for tid, j in tr.step(frame, emb_c, box_c, sc_c, fb):
                    cx, cy, w, h = box_c[j].tolist()
                    rows[m].append((frame, tid, (cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H, float(sc_c[j])))
    return rows


def main(a, detector=None):
    device = torch.device(a.device if (a.device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    ck = torch.load(a.ckpt, map_location='cpu')
    ta = argparse.Namespace(**ck['args'])
    assert 'w_assoc' in ck['args'] or 'head' not in a.modes, 'el modo head necesita un checkpoint con Associator (trackheads_v3 o posterior)'
    model = build_model(ta).to(device).eval()
    model.load_state_dict(ck['model'])
    for k in ('det_ckpt', 'det_config', 'root'):
        if getattr(a, k) is None:
            setattr(a, k, getattr(ta, k))
    if a.det_thr is None:
        a.det_thr = ta.det_thr
    if detector is None:
        from tracking.trackheads.detector import FrozenDetector
        detector = FrozenDetector(a.det_ckpt, a.det_config, ta.top_k, device, not a.no_amp)
    seqs = seq_dirs(a.root, 'val', a.seqs, a.max_seqs)
    for sd in seqs:
        t0 = time.time()
        rows = track_sequence(detector, model, sd, a, device)
        s = os.path.basename(sd)
        for m, r in rows.items():
            write_mot(os.path.join(a.out, m, f'{s}.txt'), r)
        print(f'{s} ({time.time() - t0:.0f}s): ' + ', '.join(f'{m} {len(set(x[1] for x in r))} tracks {len(r)} cajas' for m, r in rows.items()), flush=True)
    return [os.path.basename(s) for s in seqs]


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='best.pth de TrackHeads (con Associator)')
    p.add_argument('--out', required=True, help='escribe <out>/<modo>/<seq>.txt')
    p.add_argument('--modes', nargs='+', choices=['head', 'iou'], default=['head', 'iou'])
    p.add_argument('--det-ckpt', default=None, help='por defecto, el del entrenamiento')
    p.add_argument('--det-config', default=None)
    p.add_argument('--root', default=None, help='DanceTrack (usa val/); por defecto, el del entrenamiento')
    p.add_argument('--seqs', nargs='*', default=None)
    p.add_argument('--max-seqs', type=int, default=0)
    p.add_argument('--det-thr', type=float, default=None, help='score mínimo de una candidata; por defecto, el del entrenamiento (0.3)')
    p.add_argument('--birth-thresh', type=float, default=0.5)
    p.add_argument('--birth-nms', type=float, default=0.7)
    p.add_argument('--max-lost', type=int, default=30, help='frames sin asignar antes de borrar un track')
    p.add_argument('--null-bias', type=float, default=0.0, help='[head] se suma a la log-prob de "ninguna" (>0: asigna menos)')
    p.add_argument('--iou-min', type=float, default=0.3, help='[iou] IoU mínimo para asignar')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--no-amp', action='store_true')
    return p


if __name__ == '__main__':
    main(parser().parse_args())
