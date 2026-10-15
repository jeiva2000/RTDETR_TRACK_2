"""Genera un mini dataset COCO sintético (rectángulos) para probar el pipeline de entrenamiento. Solo humo, no es dato real."""
import argparse, json, os, random
import numpy as np
from PIL import Image


def build(root, split, n, seed):
    rng = random.Random(seed); os.makedirs(f'{root}/{split}', exist_ok=True)
    imgs, anns = [], []
    for i in range(n):
        a = np.full((480, 640, 3), 114, np.uint8); fn = f'{i:05d}.jpg'
        imgs.append(dict(id=i, file_name=fn, width=640, height=480))
        for _ in range(rng.randint(1, 5)):
            w, h = rng.randint(40, 150), rng.randint(80, 250); x, y = rng.randint(0, 640 - w), rng.randint(0, 480 - h)
            a[y:y + h, x:x + w] = [rng.randint(0, 255) for _ in range(3)]
            anns.append(dict(id=len(anns), image_id=i, category_id=0, bbox=[x, y, w, h], area=w * h, iscrowd=0))
        Image.fromarray(a).save(f'{root}/{split}/{fn}')
    os.makedirs(f'{root}/annotations', exist_ok=True)
    json.dump(dict(images=imgs, annotations=anns, categories=[dict(id=0, name='person')]), open(f'{root}/annotations/{split}.json', 'w'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', default=os.path.join(os.environ.get('DATA_ROOT', '.'), 'smoke_coco'))
    p.add_argument('--seed', type=int, default=int(os.environ.get('SEED', 42))); a = p.parse_args()
    build(a.out, 'train', 64, a.seed); build(a.out, 'val', 16, a.seed + 1); print('OK', a.out)
