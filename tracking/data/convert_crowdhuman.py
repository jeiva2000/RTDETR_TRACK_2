"""CrowdHuman (.odgt) -> COCO de 1 clase (person), usando las cajas de cuerpo completo (fbox).

- tag 'person' con extra.ignore == 0  -> iscrowd=0
- tag 'mask' o extra.ignore == 1      -> iscrowd=1 (marca de ignore; COCO eval y el dataloader de RT-DETR las ignoran)
- Solo se incluyen imágenes que existen en disco (permite probar con una muestra).
file_name es relativo a --root (img_folder = CROWDHUMAN_DIR en el config del detector).
"""
import argparse, json, os
from PIL import Image


def convert(odgt, root, img_dir, max_images):
    images, anns = [], []
    n_missing = n_nofbox = 0
    with open(odgt) as f:
        for line in f:
            if max_images and len(images) >= max_images:
                break
            r = json.loads(line)
            rel = os.path.join(img_dir, f"{r['ID']}.jpg")
            fp = os.path.join(root, rel)
            if not os.path.isfile(fp):
                n_missing += 1; continue
            W, H = Image.open(fp).size
            iid = len(images)
            images.append(dict(id=iid, file_name=rel, width=W, height=H))
            for g in r['gtboxes']:
                if 'fbox' not in g:
                    n_nofbox += 1; continue
                x, y, w, h = g['fbox']
                x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
                if x1 - x0 < 1 or y1 - y0 < 1:
                    continue
                ignore = g.get('tag') != 'person' or g.get('extra', {}).get('ignore', 0) == 1
                anns.append(dict(id=len(anns), image_id=iid, category_id=0, bbox=[x0, y0, x1 - x0, y1 - y0],
                                 area=(x1 - x0) * (y1 - y0), iscrowd=int(ignore)))
    print(f'{len(images)} imágenes, {len(anns)} cajas ({sum(a["iscrowd"] for a in anns)} ignore); '
          f'odgt sin imagen en disco: {n_missing}; gtboxes sin fbox: {n_nofbox}')
    return dict(images=images, annotations=anns, categories=[dict(id=0, name='person')])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=os.environ.get('CROWDHUMAN_DIR'), help='carpeta con Images/ y annotation_train.odgt')
    p.add_argument('--odgt', default=None)
    p.add_argument('--img-dir', default='Images')
    p.add_argument('--out', default=os.path.join(os.environ['COCO_DIR'], 'crowdhuman') if os.environ.get('COCO_DIR') else None)
    p.add_argument('--max-images', type=int, default=0, help='0 = todas las que existan en disco')
    a = p.parse_args()
    assert a.root and a.out, 'define --root/--out o CROWDHUMAN_DIR y COCO_DIR'
    coco = convert(a.odgt or os.path.join(a.root, 'annotation_train.odgt'), a.root, a.img_dir, a.max_images)
    os.makedirs(os.path.join(a.out, 'annotations'), exist_ok=True)
    json.dump(coco, open(os.path.join(a.out, 'annotations', 'train.json'), 'w'))
