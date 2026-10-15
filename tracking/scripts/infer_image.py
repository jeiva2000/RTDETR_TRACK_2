"""Inferencia RT-DETRv2 sobre una imagen. Rutas por argumento o variables de entorno (CKPT_DIR, OUTPUT_ROOT)."""
import argparse, glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'rtdetrv2_pytorch'))
import torch, torch.nn as nn, torchvision.transforms as T
from PIL import Image, ImageDraw
from src.core import YAMLConfig


def default_image():
    root = os.environ.get('DANCETRACK_DIR') or os.path.join(os.environ.get('DATA_ROOT', '.'), 'DanceTrack')
    fs = sorted(glob.glob(os.path.join(root, 'val', '*', 'img1', '*.jpg')))
    if not fs:
        sys.exit(f'No hay imágenes en {root}/val; define DANCETRACK_DIR o pasa --image')
    return fs[len(fs) // 2]


def main(a):
    a.image = a.image or default_image()
    print('imagen:', a.image)
    cfg = YAMLConfig(a.config, resume=a.ckpt)
    ck = torch.load(a.ckpt, map_location='cpu')
    cfg.model.load_state_dict(ck['ema']['module'] if 'ema' in ck else ck['model'])

    class M(nn.Module):
        def __init__(s):
            super().__init__(); s.m = cfg.model.deploy(); s.p = cfg.postprocessor.deploy()
        def forward(s, x, sz): return s.p(s.m(x), sz)

    dev = a.device
    model = M().to(dev).eval()
    im = Image.open(a.image).convert('RGB'); w, h = im.size
    x = T.Compose([T.Resize((640, 640)), T.ToTensor()])(im)[None].to(dev)
    with torch.no_grad():
        labels, boxes, scores = model(x, torch.tensor([[w, h]]).to(dev))
    keep = scores[0] > a.thr
    d = ImageDraw.Draw(im)
    for l, b, s in zip(labels[0][keep], boxes[0][keep], scores[0][keep]):
        d.rectangle(b.tolist(), outline='red'); d.text((b[0], b[1]), f'{l.item()} {s.item():.2f}', fill='blue')
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True); im.save(a.out)
    print(f'{keep.sum().item()} detecciones > {a.thr}; guardado en {a.out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml')
    p.add_argument('--ckpt', default=os.path.join(os.environ.get('CKPT_DIR', '.'), 'rtdetrv2_r50vd_6x_coco_ema.pth'))
    p.add_argument('--image', default=None, help='por defecto: un frame de DanceTrack val')
    p.add_argument('--out', default=os.path.join(os.environ.get('OUTPUT_ROOT', '.'), 'infer', 'result.jpg'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--thr', type=float, default=0.6)
    main(p.parse_args())
