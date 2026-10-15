"""DanceTrack (formato MOT) -> COCO de 1 clase (person) + archivo aparte con track IDs.

Salidas en --out:
  annotations/<split>.json          COCO estándar (images, annotations, categories)
  annotations/<split>_tracks.json   {"annotations": [{ann_id, image_id, seq, frame, track_id}, ...]}
file_name es relativo a --root (img_folder = DANCETRACK_DIR en el config del detector).
Rutas por argumento o variables de entorno (DANCETRACK_DIR, COCO_DIR).
"""
import argparse, configparser, json, os


def read_seqinfo(seq_dir):
    c = configparser.ConfigParser(); c.read(os.path.join(seq_dir, 'seqinfo.ini'))
    s = c['Sequence']
    return int(s['seqLength']), int(s['imWidth']), int(s['imHeight']), s.get('imExt', '.jpg')


def read_gt(path):
    by_frame = {}
    with open(path) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 6:
                continue
            fr, tid = int(p[0]), int(p[1])
            x, y, w, h = map(float, p[2:6])
            conf = float(p[6]) if len(p) > 6 else 1.0
            if conf == 0:  # marca MOT de "ignorar"
                continue
            by_frame.setdefault(fr, []).append((tid, x, y, w, h))
    return by_frame


def convert_split(root, dirs, stride, max_seqs, split):
    images, anns, tracks = [], [], []
    n_seq = 0
    for d in dirs:
        base = os.path.join(root, d)
        for seq in sorted(os.listdir(base)):
            sd = os.path.join(base, seq)
            if not os.path.isfile(os.path.join(sd, 'seqinfo.ini')):
                continue
            if max_seqs and n_seq >= max_seqs:
                break
            gt_path = os.path.join(sd, 'gt', 'gt.txt')
            if not os.path.isfile(gt_path):
                print(f'[AVISO] {sd} sin gt/gt.txt, se omite'); continue
            n_seq += 1
            length, W, H, ext = read_seqinfo(sd)
            gt = read_gt(gt_path)
            for fr in range(1, length + 1, stride):
                iid = len(images)
                images.append(dict(id=iid, file_name=f'{d}/{seq}/img1/{fr:08d}{ext}', width=W, height=H, seq=seq, frame=fr))
                for tid, x, y, w, h in gt.get(fr, []):
                    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
                    if x1 - x0 < 1 or y1 - y0 < 1:
                        continue
                    aid = len(anns)
                    anns.append(dict(id=aid, image_id=iid, category_id=0, bbox=[x0, y0, x1 - x0, y1 - y0],
                                     area=(x1 - x0) * (y1 - y0), iscrowd=0))
                    tracks.append(dict(ann_id=aid, image_id=iid, seq=seq, frame=fr, track_id=tid))
    coco = dict(images=images, annotations=anns, categories=[dict(id=0, name='person')])
    print(f'{split}: {n_seq} secuencias, {len(images)} imágenes, {len(anns)} cajas')
    return coco, dict(annotations=tracks)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=os.environ.get('DANCETRACK_DIR'), help='carpeta con train1/ train2/ val/')
    p.add_argument('--out', default=os.path.join(os.environ.get('COCO_DIR', ''), 'dancetrack') if os.environ.get('COCO_DIR') else None)
    p.add_argument('--train-dirs', nargs='+', default=['train1', 'train2'])
    p.add_argument('--val-dirs', nargs='+', default=['val'])
    p.add_argument('--stride-train', type=int, default=2)
    p.add_argument('--stride-val', type=int, default=1)
    p.add_argument('--max-seqs', type=int, default=0, help='0 = todas; >0 limita secuencias por split (muestra local)')
    p.add_argument('--splits', nargs='+', default=['train', 'val'])
    a = p.parse_args()
    assert a.root and a.out, 'define --root/--out o las variables DANCETRACK_DIR y COCO_DIR'
    os.makedirs(os.path.join(a.out, 'annotations'), exist_ok=True)
    cfg = dict(train=(a.train_dirs, a.stride_train), val=(a.val_dirs, a.stride_val))
    for s in a.splits:
        coco, tr = convert_split(a.root, cfg[s][0], cfg[s][1], a.max_seqs, s)
        json.dump(coco, open(os.path.join(a.out, 'annotations', f'{s}.json'), 'w'))
        json.dump(tr, open(os.path.join(a.out, 'annotations', f'{s}_tracks.json'), 'w'))
