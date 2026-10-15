"""Verifica un dataset COCO: conteos por split y N imágenes aleatorias con sus cajas dibujadas.

Uso:
  python tracking/data/verify_coco.py --ann $COCO_DIR/dancetrack/annotations/train.json $COCO_DIR/dancetrack/annotations/val.json \
      --img-root $DANCETRACK_DIR --out $OUTPUT_ROOT/verify/dancetrack
--ann admite varios JSON; --img-root es la carpeta a la que son relativos los file_name.
"""
import argparse, json, os, random
from collections import Counter
from PIL import Image, ImageDraw


def main(a):
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    for path in a.ann:
        name = os.path.splitext(os.path.basename(path))[0]
        d = json.load(open(path))
        imgs, anns = d['images'], d['annotations']
        per_img = Counter(x['image_id'] for x in anns)
        n_crowd = sum(x.get('iscrowd', 0) for x in anns)
        bad = sum(1 for x in anns if x['bbox'][2] <= 0 or x['bbox'][3] <= 0)
        n_empty = sum(1 for i in imgs if per_img[i['id']] == 0)
        seqs = {i['seq'] for i in imgs if 'seq' in i}
        print(f'== {name} ({path})')
        print(f'   imágenes: {len(imgs)}  cajas: {len(anns)}  cajas/imagen: {len(anns) / max(len(imgs), 1):.2f}'
              f'  sin cajas: {n_empty}  iscrowd/ignore: {n_crowd}  cajas inválidas: {bad}')
        print(f'   categorías: {[(c["id"], c["name"]) for c in d["categories"]]}' + (f'  secuencias: {len(seqs)}' if seqs else ''))
        by_img = {}
        for x in anns:
            by_img.setdefault(x['image_id'], []).append(x)
        for i in rng.sample(imgs, min(a.n, len(imgs))):
            fp = os.path.join(a.img_root, i['file_name'])
            if not os.path.isfile(fp):
                print(f'   [FALTA] {fp}'); continue
            im = Image.open(fp).convert('RGB'); dr = ImageDraw.Draw(im)
            for x in by_img.get(i['id'], []):
                bx, by, w, h = x['bbox']
                dr.rectangle([bx, by, bx + w, by + h], outline='yellow' if x.get('iscrowd') else 'red', width=3)
            out = os.path.join(a.out, f'{name}_{i["id"]}.jpg'); im.save(out)
        print(f'   {min(a.n, len(imgs))} imágenes guardadas en {a.out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ann', nargs='+', required=True)
    p.add_argument('--img-root', default=os.environ.get('DANCETRACK_DIR'))
    p.add_argument('--out', default=os.path.join(os.environ.get('OUTPUT_ROOT', '.'), 'verify'))
    p.add_argument('--n', type=int, default=10)
    p.add_argument('--seed', type=int, default=int(os.environ.get('SEED', 42)))
    a = p.parse_args()
    assert a.img_root, 'define --img-root o DANCETRACK_DIR'
    main(a)
