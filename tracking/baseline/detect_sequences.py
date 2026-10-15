"""Ejecuta el detector (1 clase) sobre los frames de secuencias DanceTrack y guarda las detecciones en formato MOT.

Salida: <out>/<seq>.txt con filas  frame,-1,x,y,w,h,score,-1,-1,-1  (score >= --min-score), en píxeles de la imagen original.
Rutas por argumentos o variables de entorno (DANCETRACK_DIR, OUTPUT_ROOT).
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'rtdetrv2_pytorch'))
import torch, torch.nn as nn, torchvision.transforms as T
from PIL import Image
from src.core import YAMLConfig
from tracking.baseline.mot_io import read_seqinfo


class Frames(torch.utils.data.Dataset):
    def __init__(self, seq_dir, info):
        self.dir, self.n, self.ext = os.path.join(seq_dir, 'img1'), info['length'], info['ext']
        self.tf = T.Compose([T.Resize((640, 640)), T.ToTensor()])

    def __len__(self): return self.n

    def __getitem__(self, i):
        im = Image.open(os.path.join(self.dir, f'{i + 1:08d}{self.ext}')).convert('RGB')
        return self.tf(im), torch.tensor([im.size[0], im.size[1]]), i + 1


def load_model(config, ckpt, device):
    cfg = YAMLConfig(config, PResNet=dict(pretrained=False))  # no hace falta bajar pesos ImageNet: se cargan del checkpoint
    ck = torch.load(ckpt, map_location='cpu')
    cfg.model.load_state_dict(ck['ema']['module'] if 'ema' in ck else ck['model'])

    class M(nn.Module):
        def __init__(s):
            super().__init__(); s.m = cfg.model.deploy(); s.p = cfg.postprocessor.deploy()
        def forward(s, x, sz): return s.p(s.m(x), sz)
    return M().to(device).eval()


@torch.no_grad()
def main(a):
    device = a.device
    model = load_model(a.config, a.ckpt, device)
    seqs = a.seqs or sorted(d for d in os.listdir(a.seq_root) if os.path.isfile(os.path.join(a.seq_root, d, 'seqinfo.ini')))
    if a.max_seqs:
        seqs = seqs[:a.max_seqs]
    os.makedirs(a.out, exist_ok=True)
    for s in seqs:
        info = read_seqinfo(os.path.join(a.seq_root, s))
        dl = torch.utils.data.DataLoader(Frames(os.path.join(a.seq_root, s), info), batch_size=a.batch,
                                         num_workers=a.num_workers, shuffle=False)
        rows = []
        for x, sz, fr in dl:
            _, boxes, scores = model(x.to(device), sz.to(device))
            boxes, scores = boxes.cpu().numpy(), scores.cpu().numpy()
            for b in range(len(fr)):
                k = scores[b] >= a.min_score
                for (x1, y1, x2, y2), sc in zip(boxes[b][k], scores[b][k]):
                    rows.append((int(fr[b]), -1, x1, y1, x2 - x1, y2 - y1, sc, -1, -1, -1))
        np.savetxt(os.path.join(a.out, f'{s}.txt'), np.array(rows).reshape(-1, 10), delimiter=',',
                   fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,%d,%d,%d')
        print(f'{s}: {info["length"]} frames, {len(rows)} detecciones (score >= {a.min_score})', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'configs', 'detector_r50vd_person.yml'))
    p.add_argument('--ckpt', required=True, help='best.pth del detector entrenado')
    p.add_argument('--seq-root', default=os.environ.get('DANCETRACK_DIR', '') and os.path.join(os.environ.get('DANCETRACK_DIR', ''), 'val'))
    p.add_argument('--out', required=True)
    p.add_argument('--seqs', nargs='*', default=None)
    p.add_argument('--max-seqs', type=int, default=0)
    p.add_argument('--min-score', type=float, default=0.1)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    main(p.parse_args())
