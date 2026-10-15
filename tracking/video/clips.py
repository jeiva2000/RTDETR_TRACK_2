"""Muestreo de clips de T frames con stride aleatorio y frame skipping, y datasets de clips.

- CachedClipDataset: lee las features cacheadas por cache_features.py (rápido; detector congelado).
- ImageClipDataset : devuelve imágenes + GT del clip para extraer las queries en línea con QueryExtractor.
Los dos usan el mismo muestreo de índices.
"""
import os, random
import torch

from tracking.baseline.mot_io import read_seqinfo
from tracking.video.frames import load_gt, load_frame


def sample_clip(n_frames, T, max_stride=1, p_skip=0.0, max_skip=0, rng=random):
    """Frames (1-indexados) de un clip: stride base s ~ U{1..max_stride}; en cada paso, con prob p_skip,
    se saltan además U{1..max_skip} frames (simula detecciones perdidas / oclusiones largas).
    Si la secuencia es corta, se reduce el stride hasta que quepa."""
    s = rng.randint(1, max(1, max_stride))
    steps = [s + (rng.randint(1, max_skip) if max_skip and rng.random() < p_skip else 0) for _ in range(T - 1)]
    while sum(steps) >= n_frames and max(steps) > 1:
        steps = [max(1, x - 1) for x in steps]
    span = sum(steps)
    if span >= n_frames:
        raise ValueError(f'secuencia de {n_frames} frames demasiado corta para T={T}')
    start = rng.randint(1, n_frames - span)
    fr = [start]
    for x in steps:
        fr.append(fr[-1] + x)
    return fr


class _ClipBase(torch.utils.data.Dataset):
    def __init__(self, seqs, lengths, T, samples_per_epoch, max_stride, p_skip, max_skip, seed):
        self.seqs, self.lengths, self.T, self.n = seqs, lengths, T, samples_per_epoch
        self.kw = dict(max_stride=max_stride, p_skip=p_skip, max_skip=max_skip)
        self.seed = seed

    def __len__(self): return self.n

    def _pick(self, i):
        rng = random.Random(self.seed * 1000003 + i)
        s = rng.choice(self.seqs)
        return s, sample_clip(self.lengths[s], self.T, rng=rng, **self.kw)


class CachedClipDataset(_ClipBase):
    """Cada item: dict con emb (T,K,256), boxes (T,K,4), scores (T,K), track_id (T,K; -1 fondo), match_iou (T,K),
    frames (T,), seq (str)."""

    def __init__(self, cache_dir, seqs=None, T=10, samples_per_epoch=1000, max_stride=1, p_skip=0.0, max_skip=0, seed=0):
        seqs = seqs or sorted(f[:-3] for f in os.listdir(cache_dir) if f.endswith('.pt'))
        self.data = {s: torch.load(os.path.join(cache_dir, f'{s}.pt')) for s in seqs}
        super().__init__(seqs, {s: d['emb'].shape[0] for s, d in self.data.items()}, T, samples_per_epoch,
                         max_stride, p_skip, max_skip, seed)

    def __getitem__(self, i):
        s, fr = self._pick(i)
        d, ix = self.data[s], torch.tensor(fr) - 1
        return dict(emb=d['emb'][ix].float(), boxes=d['boxes'][ix], scores=d['scores'][ix],
                    track_id=d['track_id'][ix].long(), match_iou=d['match_iou'][ix], frames=torch.tensor(fr), seq=s)


class ImageClipDataset(_ClipBase):
    """Cada item: images (T,3,640,640), gts (lista T de (N,5)), frames, seq. La extracción se hace en el bucle de entrenamiento."""

    def __init__(self, seq_root, seqs=None, T=10, samples_per_epoch=1000, max_stride=1, p_skip=0.0, max_skip=0, seed=0):
        seqs = seqs or sorted(d for d in os.listdir(seq_root) if os.path.isfile(os.path.join(seq_root, d, 'seqinfo.ini')))
        self.root = seq_root
        self.info = {s: read_seqinfo(os.path.join(seq_root, s)) for s in seqs}
        self.gt = {s: load_gt(os.path.join(seq_root, s), self.info[s]) for s in seqs}
        super().__init__(seqs, {s: self.info[s]['length'] for s in seqs}, T, samples_per_epoch,
                         max_stride, p_skip, max_skip, seed)

    def __getitem__(self, i):
        s, fr = self._pick(i)
        d = os.path.join(self.root, s)
        return dict(images=torch.stack([load_frame(d, self.info[s], f) for f in fr]),
                    gts=[self.gt[s].get(f, torch.zeros(0, 5)) for f in fr], frames=torch.tensor(fr), seq=s)


def collate_images(batch):
    return dict(images=torch.stack([b['images'] for b in batch]), gts=[b['gts'] for b in batch],
                frames=torch.stack([b['frames'] for b in batch]), seq=[b['seq'] for b in batch])
