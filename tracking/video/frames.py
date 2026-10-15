"""Lectura de frames y GT de secuencias DanceTrack (formato MOT) con el mismo preprocesado que el detector."""
import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from tracking.baseline.mot_io import read_mot, read_seqinfo

TF = T.Compose([T.Resize((640, 640)), T.ToTensor()])


def load_gt(seq_dir, info):
    """{frame: tensor (N,5) [track_id, cx, cy, w, h] normalizados a la imagen}"""
    a = read_mot(os.path.join(seq_dir, 'gt', 'gt.txt'))
    a = a[a[:, 6] != 0]
    W, H = info['width'], info['height']
    out = {}
    for f in np.unique(a[:, 0]).astype(int):
        r = a[a[:, 0] == f]
        out[f] = torch.tensor(np.stack([r[:, 1], (r[:, 2] + r[:, 4] / 2) / W, (r[:, 3] + r[:, 5] / 2) / H,
                                        r[:, 4] / W, r[:, 5] / H], 1), dtype=torch.float32)
    return out


def load_frame(seq_dir, info, frame):
    return TF(Image.open(os.path.join(seq_dir, 'img1', f'{frame:08d}{info["ext"]}')).convert('RGB'))


class SequenceFrames(torch.utils.data.Dataset):
    """Todos los frames de una secuencia (para extraer y cachear)."""
    def __init__(self, seq_dir):
        self.dir, self.info = seq_dir, read_seqinfo(seq_dir)

    def __len__(self): return self.info['length']

    def __getitem__(self, i): return load_frame(self.dir, self.info, i + 1), i + 1
