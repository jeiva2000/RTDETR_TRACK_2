"""Validación de la Fase 5: muestrea clips (de la caché o en línea), comprueba formas/consistencia y dibuja un clip.

  python -m tracking.video.check_clips --cache <dir>                      # clips desde la caché
  python -m tracking.video.check_clips --online --ckpt best.pth --split val --seqs dancetrack0004   # extracción en línea
Dibuja las queries emparejadas (color = track ID) y, en gris, las de fondo con score >= --min-score.
"""
import argparse, colorsys, os, sys
import torch
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tracking.video.clips import CachedClipDataset, ImageClipDataset, collate_images
from tracking.baseline.mot_io import read_seqinfo


def color(tid):
    r, g, b = colorsys.hsv_to_rgb((tid * 0.618) % 1, 0.9, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_clip(seq_dir, frames, boxes, tids, scores, out_dir, min_score):
    info = read_seqinfo(seq_dir); W, H = info['width'], info['height']
    os.makedirs(out_dir, exist_ok=True)
    for t, f in enumerate(frames.tolist()):
        im = Image.open(os.path.join(seq_dir, 'img1', f'{f:08d}{info["ext"]}')).convert('RGB'); d = ImageDraw.Draw(im)
        for b, tid, sc in zip(boxes[t], tids[t], scores[t]):
            if tid < 0 and sc < min_score:
                continue
            cx, cy, w, h = b.tolist()
            xy = [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]
            d.rectangle(xy, outline=color(int(tid)) if tid >= 0 else (128, 128, 128), width=3 if tid >= 0 else 1)
            if tid >= 0:
                d.text((xy[0] + 3, xy[1] + 3), str(int(tid)), fill=color(int(tid)))
        im.save(os.path.join(out_dir, f'{t:02d}_frame{f:06d}.jpg'))


def summarize(tids, frames):
    ids = [set(x[x >= 0].tolist()) for x in tids]
    persist = len(set.intersection(*ids)) if ids else 0
    print(f'  frames {frames.tolist()}')
    print(f'  queries emparejadas por frame: {[len(i) for i in ids]}  | IDs presentes en todo el clip: {persist}'
          f'  | IDs distintos: {len(set.union(*ids)) if ids else 0}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache', default=None)
    p.add_argument('--online', action='store_true')
    p.add_argument('--ckpt'); p.add_argument('--split', default='val'); p.add_argument('--seqs', nargs='*', default=None)
    p.add_argument('--root', default=os.environ.get('DANCETRACK_DIR'))
    p.add_argument('--T', type=int, default=10); p.add_argument('--max-stride', type=int, default=3)
    p.add_argument('--p-skip', type=float, default=0.2); p.add_argument('--max-skip', type=int, default=5)
    p.add_argument('--n', type=int, default=3); p.add_argument('--min-score', type=float, default=0.5)
    p.add_argument('--top-k', type=int, default=100); p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', default=os.path.join(os.environ.get('OUTPUT_ROOT', '.'), 'check_clips'))
    a = p.parse_args()
    kw = dict(T=a.T, samples_per_epoch=a.n, max_stride=a.max_stride, p_skip=a.p_skip, max_skip=a.max_skip, seed=0)
    split_dirs = {'train': ['train1', 'train2'], 'val': ['val']}[a.split]
    find_seq = lambda s: next(os.path.join(a.root, d, s) for d in split_dirs if os.path.isdir(os.path.join(a.root, d, s)))
    if a.online:
        from tracking.video.extractor import QueryExtractor
        ex = QueryExtractor(a.ckpt, top_k=a.top_k, device=a.device)
        seq_root = os.path.join(a.root, split_dirs[0]) if len(split_dirs) == 1 else None
        assert seq_root, '--online con --split train: usa --root apuntando a train1 o train2 directamente'
        ds = ImageClipDataset(seq_root, a.seqs, **kw)
        for i in range(len(ds)):
            b = collate_images([ds[i]])
            q = ex(b['images'][0]); tids, _ = ex.match(q, b['gts'][0])
            print(f'clip {i} ({b["seq"][0]}): emb {tuple(q["emb"].shape)} boxes {tuple(q["boxes"].shape)}')
            summarize(tids, b['frames'][0])
            draw_clip(find_seq(b['seq'][0]), b['frames'][0], q['boxes'].cpu(), tids, q['scores'].cpu(), os.path.join(a.out, f'clip{i}'), a.min_score)
    else:
        ds = CachedClipDataset(a.cache, a.seqs, **kw)
        for i in range(len(ds)):
            c = ds[i]
            print(f'clip {i} ({c["seq"]}): emb {tuple(c["emb"].shape)} boxes {tuple(c["boxes"].shape)} '
                  f'fondo {(c["track_id"] < 0).float().mean():.2f}')
            summarize(c['track_id'], c['frames'])
            draw_clip(find_seq(c['seq']), c['frames'], c['boxes'], c['track_id'], c['scores'], os.path.join(a.out, f'clip{i}'), a.min_score)
    print('imágenes en', a.out)
