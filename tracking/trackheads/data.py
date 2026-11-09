"""Pares de frames (t, t+k) de DanceTrack para TrackHeads: imágenes con el preprocesado del detector (640×640, [0,1]) y GT de los dos frames.
k ~ U{k_min..k_max}. En train cada índice se re-muestrea por época (set_epoch) con un RNG derivado de (seed, época, índice): reproducible e
independiente del número de workers. En val la lista de pares es fija (misma semilla -> mismos pares en todas las épocas y ejecuciones)."""
import os, random

import torch

from tracking.baseline.mot_io import read_seqinfo
from tracking.video.frames import load_frame, load_gt

SPLIT_DIRS = {'train': ['train1', 'train2'], 'val': ['val']}


def seq_dirs(root, split, seqs=None, max_seqs=0):
    out = []
    for d in SPLIT_DIRS[split]:
        base = os.path.join(root, d)
        if not os.path.isdir(base):
            continue
        for s in sorted(os.listdir(base)):
            if os.path.isfile(os.path.join(base, s, 'seqinfo.ini')) and (not seqs or s in seqs):
                out.append(os.path.join(base, s))
    if not out:
        raise FileNotFoundError(f'no hay secuencias de {split} en {root} ({SPLIT_DIRS[split]})')
    return out[:max_seqs] if max_seqs else out


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, k_min=1, k_max=10, samples=2000, seed=0, seqs=None, max_seqs=0, fixed=None):
        self.seqs = []
        for sd in seq_dirs(root, split, seqs, max_seqs):
            info = read_seqinfo(sd)
            if info['length'] > k_min:
                self.seqs.append((sd, info, load_gt(sd, info)))
        self.k_min, self.k_max, self.samples, self.seed, self.epoch = k_min, k_max, samples, seed, 0
        self.fixed = (split == 'val') if fixed is None else fixed
        self.pairs = [self._sample(random.Random(f'{seed}-val-{i}')) for i in range(samples)] if self.fixed else None

    def set_epoch(self, e):
        self.epoch = e

    def _sample(self, rng):
        si = rng.randrange(len(self.seqs))
        n = self.seqs[si][1]['length']
        k = rng.randint(self.k_min, min(self.k_max, n - 1))
        t = rng.randint(1, n - k)
        return si, t, k

    def __len__(self):
        return self.samples

    def __getitem__(self, i):
        si, t, k = self.pairs[i] if self.fixed else self._sample(random.Random(f'{self.seed}-{self.epoch}-{i}'))
        sd, info, gt = self.seqs[si]
        empty = torch.zeros(0, 5)
        return dict(img_t=load_frame(sd, info, t), img_tk=load_frame(sd, info, t + k),
                    gt_t=gt.get(t, empty), gt_tk=gt.get(t + k, empty), k=k, seq=os.path.basename(sd), t=t)


def collate(items):
    return dict(img_t=torch.stack([x['img_t'] for x in items]), img_tk=torch.stack([x['img_tk'] for x in items]),
                gt_t=[x['gt_t'] for x in items], gt_tk=[x['gt_tk'] for x in items],
                k=torch.tensor([x['k'] for x in items]), seq=[x['seq'] for x in items], t=torch.tensor([x['t'] for x in items]))
