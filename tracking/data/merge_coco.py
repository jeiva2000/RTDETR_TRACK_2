"""Une varios COCO (misma clase) en uno solo. Los file_name se reescriben relativos a --base,
de modo que img_folder=--base sirve para todos (p. ej. base=$DATA_ROOT con DanceTrack/ y CrowdHuman/ dentro).

Uso: python tracking/data/merge_coco.py --base $DATA_ROOT --out $COCO_DIR/detector_train/annotations/train.json \
        --inputs $COCO_DIR/dancetrack/annotations/train.json:$DANCETRACK_DIR \
                 $COCO_DIR/crowdhuman/annotations/train.json:$CROWDHUMAN_DIR
Cada entrada es <json>:<carpeta de imágenes a la que son relativos sus file_name>.
"""
import argparse, json, os


def main(a):
    base = os.path.realpath(a.base)
    images, anns = [], []
    for item in a.inputs:
        ann_path, img_root = item.rsplit(':', 1)
        rel = os.path.relpath(os.path.realpath(img_root), base)
        assert not rel.startswith('..'), f'{img_root} no está dentro de --base {base}'
        d = json.load(open(ann_path))
        assert [c['id'] for c in d['categories']] == [0], f'{ann_path}: se espera una única categoría con id 0'
        id_map = {}
        for im in d['images']:
            id_map[im['id']] = len(images)
            images.append({**im, 'id': len(images), 'file_name': os.path.join(rel, im['file_name']) if rel != '.' else im['file_name']})
        for x in d['annotations']:
            anns.append({**x, 'id': len(anns), 'image_id': id_map[x['image_id']]})
        print(f'{ann_path}: {len(d["images"])} imágenes, {len(d["annotations"])} cajas, prefijo "{rel}"')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(images=images, annotations=anns, categories=[dict(id=0, name='person')]), open(a.out, 'w'))
    print(f'-> {a.out}: {len(images)} imágenes, {len(anns)} cajas')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base', default=os.environ.get('DATA_ROOT'))
    p.add_argument('--out', required=True)
    p.add_argument('--inputs', nargs='+', required=True)
    a = p.parse_args()
    assert a.base, 'define --base o DATA_ROOT'
    main(a)
