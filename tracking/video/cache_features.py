"""Extrae y cachea, para todos los frames de cada secuencia, las top-K queries del detector congelado
y su track ID (Hungarian del detector; -1 = fondo).

Salida: <out>/<seq>.pt con
  emb (N,K,256) float16 | boxes (N,K,4) cxcywh normalizado | scores (N,K) | track_id (N,K) int32 | match_iou (N,K)
  meta: top_k, ckpt, config, min_iou, width, height
y <out>/stats.json con estadísticas del emparejamiento.
Uso: python -m tracking.video.cache_features --ckpt best.pth --split train --out $OUTPUT_ROOT/features/<run>/train [--max-seqs 2]
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tracking.video.extractor import QueryExtractor, DEFAULT_CONFIG
from tracking.video.frames import SequenceFrames, load_gt


def seq_dirs(root, split, seqs, max_seqs):
    dirs = {'train': ['train1', 'train2'], 'val': ['val']}[split]
    out = []
    for d in dirs:
        base = os.path.join(root, d)
        for s in sorted(os.listdir(base)):
            if os.path.isfile(os.path.join(base, s, 'seqinfo.ini')) and (not seqs or s in seqs):
                out.append(os.path.join(base, s))
    return out[:max_seqs] if max_seqs else out


def main(a):
    ex = QueryExtractor(a.ckpt, a.config, a.top_k, a.device)
    os.makedirs(a.out, exist_ok=True)
    stats = {}
    for sd in seq_dirs(a.root, a.split, a.seqs, a.max_seqs):
        s = os.path.basename(sd)
        ds = SequenceFrames(sd); gt = load_gt(sd, ds.info)
        dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=a.num_workers)
        E, B, S, TID, IOU = [], [], [], [], []
        n_gt = n_matched = 0
        for x, fr in dl:
            q = ex(x)
            gts = [gt.get(int(f), torch.zeros(0, 5)) for f in fr]
            tid, miou = ex.match(q, gts, a.min_iou)
            E.append(q['emb'].half().cpu()); B.append(q['boxes'].cpu()); S.append(q['scores'].cpu())
            TID.append(tid.int()); IOU.append(miou)
            n_gt += sum(len(g) for g in gts); n_matched += int((tid >= 0).sum())
        IOU = torch.cat(IOU); m = IOU[IOU > 0]
        torch.save(dict(emb=torch.cat(E), boxes=torch.cat(B), scores=torch.cat(S), track_id=torch.cat(TID), match_iou=IOU,
                        meta=dict(top_k=a.top_k, ckpt=os.path.abspath(a.ckpt), config=os.path.abspath(a.config), min_iou=a.min_iou,
                                  width=ds.info['width'], height=ds.info['height'])), os.path.join(a.out, f'{s}.pt'))
        stats[s] = dict(frames=len(ds), gt_boxes=n_gt, matched=n_matched, gt_matched_frac=n_matched / max(n_gt, 1),
                        match_iou_mean=float(m.mean()) if len(m) else 0.0,
                        match_iou_lt_0_5_frac=float((m < 0.5).float().mean()) if len(m) else 0.0)
        print(s, stats[s], flush=True)
    json.dump(stats, open(os.path.join(a.out, 'stats.json'), 'w'), indent=1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--config', default=DEFAULT_CONFIG)
    p.add_argument('--root', default=os.environ.get('DANCETRACK_DIR'))
    p.add_argument('--split', choices=['train', 'val'], required=True)
    p.add_argument('--seqs', nargs='*', default=None)
    p.add_argument('--max-seqs', type=int, default=0)
    p.add_argument('--out', required=True)
    p.add_argument('--top-k', type=int, default=100)
    p.add_argument('--min-iou', type=float, default=0.0, help='0 = Hungarian puro (plan); >0 descarta matches con IoU menor')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    main(p.parse_args())
