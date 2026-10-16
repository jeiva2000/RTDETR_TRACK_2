"""Muestreo de clips de T frames con stride aleatorio y frame skipping, y datasets de clips.

- CachedClipDataset: lee las features cacheadas por cache_features.py (rápido; detector congelado).
- ImageClipDataset : devuelve imágenes + GT del clip para extraer las queries en línea con QueryExtractor.
Los dos usan el mismo muestreo de índices.

Cruces (opcional, CachedClipDataset con gt_root): un cruce es un par de cajas GT con IoU > cross_iou en un frame.
Cada clip incluye 'cross' = por frame, la lista de pares (id_a, id_b) que se cruzan. Con p_cross > 0, una fracción
p_cross de los clips se centra en un frame de cruce elegido al azar (sobremuestreo de casos difíciles).
"""
import os, random
import numpy as np
import torch

from tracking.baseline.mot_io import read_seqinfo, read_mot, by_frame
from tracking.baseline.crossings import iou_matrix
from tracking.video.frames import load_gt, load_frame


def sample_clip(n_frames, T, max_stride=1, p_skip=0.0, max_skip=0, rng=random, anchor=None):
    """Frames (1-indexados) de un clip: stride base s ~ U{1..max_stride}; en cada paso, con prob p_skip,
    se saltan además U{1..max_skip} frames (simula detecciones perdidas / oclusiones largas).
    Si la secuencia es corta, se reduce el stride hasta que quepa.
    anchor: si se da, el clip contiene ese frame en una posición aleatoria (salvo que choque con el borde
    de la secuencia, en cuyo caso se desplaza lo mínimo para caber)."""
    s = rng.randint(1, max(1, max_stride))
    steps = [s + (rng.randint(1, max_skip) if max_skip and rng.random() < p_skip else 0) for _ in range(T - 1)]
    while sum(steps) >= n_frames and max(steps) > 1:
        steps = [max(1, x - 1) for x in steps]
    span = sum(steps)
    if span >= n_frames:
        raise ValueError(f'secuencia de {n_frames} frames demasiado corta para T={T}')
    if anchor is None:
        start = rng.randint(1, n_frames - span)
    else:
        k = rng.randint(0, T - 1)
        start = min(max(1, anchor - sum(steps[:k])), n_frames - span)
    fr = [start]
    for x in steps:
        fr.append(fr[-1] + x)
    return fr


def crossing_pairs(seq_dir, iou_thr):
    """{frame: [(id_a, id_b), ...]} con los pares de cajas GT de la secuencia con IoU > iou_thr."""
    gt = read_mot(os.path.join(seq_dir, 'gt', 'gt.txt'))
    out = {}
    for f, rows in by_frame(gt[gt[:, 6] != 0]).items():
        if len(rows) < 2:
            continue
        m = np.triu(iou_matrix(rows[:, 2:6], rows[:, 2:6]), 1)
        a, b = np.where(m > iou_thr)
        if len(a):
            out[int(f)] = [(int(rows[i, 1]), int(rows[j, 1])) for i, j in zip(a, b)]
    return out


def find_seq_dir(gt_root, seq):
    for d in ('train1', 'train2', 'val', ''):
        p = os.path.join(gt_root, d, seq)
        if os.path.isfile(os.path.join(p, 'seqinfo.ini')):
            return p
    raise FileNotFoundError(f'{seq} no está en {gt_root}/(train1|train2|val)')


class _ClipBase(torch.utils.data.Dataset):
    def __init__(self, seqs, lengths, T, samples_per_epoch, max_stride, p_skip, max_skip, seed, cross=None, p_cross=0.0):
        self.seqs, self.lengths, self.T, self.n = seqs, lengths, T, samples_per_epoch
        self.kw = dict(max_stride=max_stride, p_skip=p_skip, max_skip=max_skip)
        self.seed = seed
        self.cross = cross or {}
        self.p_cross = p_cross
        self.cross_frames = [(s, f) for s in seqs for f in sorted(self.cross.get(s, {}))]
        assert p_cross == 0 or self.cross_frames, 'p_cross > 0 pero no hay cruces (¿falta gt_root?)'

    def __len__(self): return self.n

    def _pick(self, i):
        rng = random.Random(self.seed * 1000003 + i)
        if self.p_cross > 0 and rng.random() < self.p_cross:
            s, f = rng.choice(self.cross_frames)
            return s, sample_clip(self.lengths[s], self.T, rng=rng, anchor=f, **self.kw)
        s = rng.choice(self.seqs)
        return s, sample_clip(self.lengths[s], self.T, rng=rng, **self.kw)

    def _cross(self, s, fr):
        c = self.cross.get(s, {})
        return [c.get(f, []) for f in fr]


class CachedClipDataset(_ClipBase):
    """Cada item: dict con emb (T,K,256), boxes (T,K,4), scores (T,K), track_id (T,K; -1 fondo), match_iou (T,K),
    frames (T,), seq (str), cross (lista T de pares GT que se cruzan; vacía si no hay gt_root)."""

    def __init__(self, cache_dir, seqs=None, T=10, samples_per_epoch=1000, max_stride=1, p_skip=0.0, max_skip=0, seed=0,
                 gt_root=None, cross_iou=0.5, p_cross=0.0):
        seqs = seqs or sorted(f[:-3] for f in os.listdir(cache_dir) if f.endswith('.pt'))
        self.data = {s: torch.load(os.path.join(cache_dir, f'{s}.pt')) for s in seqs}
        cross = {s: crossing_pairs(find_seq_dir(gt_root, s), cross_iou) for s in seqs} if gt_root else None
        super().__init__(seqs, {s: d['emb'].shape[0] for s, d in self.data.items()}, T, samples_per_epoch,
                         max_stride, p_skip, max_skip, seed, cross, p_cross)

    def __getitem__(self, i):
        s, fr = self._pick(i)
        d, ix = self.data[s], torch.tensor(fr) - 1
        return dict(emb=d['emb'][ix].float(), boxes=d['boxes'][ix], scores=d['scores'][ix],
                    track_id=d['track_id'][ix].long(), match_iou=d['match_iou'][ix], frames=torch.tensor(fr), seq=s,
                    cross=self._cross(s, fr))


class ImageClipDataset(_ClipBase):
    """Cada item: images (T,3,640,640), gts (lista T de (N,5)), frames, seq. La extracción se hace en el bucle de entrenamiento.
    seq_root: carpeta con secuencias, o lista de carpetas (p. ej. train1 y train2). flip=True voltea el clip entero con prob. 0.5."""

    def __init__(self, seq_root, seqs=None, T=10, samples_per_epoch=1000, max_stride=1, p_skip=0.0, max_skip=0, seed=0, flip=False):
        roots = [seq_root] if isinstance(seq_root, str) else list(seq_root)
        self.dirs = {}
        for r in roots:
            for d in sorted(os.listdir(r)):
                if os.path.isfile(os.path.join(r, d, 'seqinfo.ini')) and (not seqs or d in seqs):
                    self.dirs[d] = os.path.join(r, d)
        seqs = sorted(self.dirs)
        self.flip = flip
        self.info = {s: read_seqinfo(self.dirs[s]) for s in seqs}
        self.gt = {s: load_gt(self.dirs[s], self.info[s]) for s in seqs}
        super().__init__(seqs, {s: self.info[s]['length'] for s in seqs}, T, samples_per_epoch,
                         max_stride, p_skip, max_skip, seed)

    def __getitem__(self, i):
        s, fr = self._pick(i)
        images = torch.stack([load_frame(self.dirs[s], self.info[s], f) for f in fr])
        gts = [self.gt[s].get(f, torch.zeros(0, 5)) for f in fr]
        if self.flip and random.Random(self.seed * 999983 + i).random() < 0.5:
            images = images.flip(-1)
            gts = [torch.cat([g[:, :1], 1 - g[:, 1:2], g[:, 2:]], 1) if len(g) else g for g in gts]
        return dict(images=images, gts=gts, frames=torch.tensor(fr), seq=s)


def collate_images(batch):
    return dict(images=torch.stack([b['images'] for b in batch]), gts=[b['gts'] for b in batch],
                frames=torch.stack([b['frames'] for b in batch]), seq=[b['seq'] for b in batch])
