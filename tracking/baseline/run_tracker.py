"""Ejecuta ByteTrack u OC-SORT (código oficial vendorizado en tracking/third_party) sobre detecciones en formato MOT.

Entrada : <dets>/<seq>.txt      (frame,-1,x,y,w,h,score,...)   -> salida de detect_sequences.py
Salida  : <out>/<seq>.txt       (frame,id,x,y,w,h,1,-1,-1,-1)  -> listo para TrackEval
Los parámetros por defecto son los de los repos oficiales (no ajustados a este detector).
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tracking.baseline.mot_io import read_mot, read_seqinfo, by_frame, write_mot


def make_tracker(name, a):
    if name == 'bytetrack':
        from tracking.third_party.bytetrack import BYTETracker
        from tracking.third_party.bytetrack.basetrack import BaseTrack
        BaseTrack._count = 0
        args = argparse.Namespace(track_thresh=a.track_thresh, track_buffer=a.track_buffer,
                                  match_thresh=a.match_thresh, mot20=False)
        return BYTETracker(args, frame_rate=30)  # igual que el repo oficial (frame_rate por defecto)
    if name == 'ocsort':
        from tracking.third_party.ocsort import OCSort
        return OCSort(det_thresh=a.track_thresh, max_age=a.track_buffer, min_hits=a.min_hits,
                      iou_threshold=a.iou_thresh, delta_t=a.deltat, asso_func='iou', inertia=a.inertia, use_byte=False)
    raise ValueError(name)


def run_seq(name, a, dets, info):
    tr = make_tracker(name, a)
    H, W = info['height'], info['width']
    frames = by_frame(dets)
    rows = []
    for fr in range(1, info['length'] + 1):
        d = frames.get(fr)
        if d is None or len(d) == 0:
            arr = np.zeros((0, 5))
        else:  # x,y,w,h -> x1,y1,x2,y2,score
            arr = np.stack([d[:, 2], d[:, 3], d[:, 2] + d[:, 4], d[:, 3] + d[:, 5], d[:, 6]], 1).astype(np.float64)
        if name == 'bytetrack':
            for t in tr.update(arr, (H, W), (H, W)):
                x, y, w, h = t.tlwh
                rows.append((fr, t.track_id, x, y, w, h, 1.0))
        else:
            for t in tr.update(arr, (H, W), (H, W)):
                x1, y1, x2, y2, tid = t[:5]
                rows.append((fr, tid, x1, y1, x2 - x1, y2 - y1, 1.0))
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tracker', choices=['bytetrack', 'ocsort'], required=True)
    p.add_argument('--dets', required=True)
    p.add_argument('--seq-root', default=os.environ.get('DANCETRACK_DIR', '') and os.path.join(os.environ.get('DANCETRACK_DIR', ''), 'val'),
                   help='carpeta con <seq>/seqinfo.ini (por defecto $DANCETRACK_DIR/val)')
    p.add_argument('--out', required=True)
    p.add_argument('--track-thresh', type=float, default=0.6)
    p.add_argument('--track-buffer', type=int, default=30)
    p.add_argument('--match-thresh', type=float, default=0.9, help='solo ByteTrack')
    p.add_argument('--min-hits', type=int, default=3, help='solo OC-SORT')
    p.add_argument('--iou-thresh', type=float, default=0.3, help='solo OC-SORT')
    p.add_argument('--deltat', type=int, default=3, help='solo OC-SORT')
    p.add_argument('--inertia', type=float, default=0.2, help='solo OC-SORT')
    a = p.parse_args()
    seqs = sorted(f[:-4] for f in os.listdir(a.dets) if f.endswith('.txt'))
    for s in seqs:
        rows = run_seq(a.tracker, a, read_mot(os.path.join(a.dets, f'{s}.txt')), read_seqinfo(os.path.join(a.seq_root, s)))
        write_mot(os.path.join(a.out, f'{s}.txt'), rows)
        print(f'{a.tracker} {s}: {len(set(r[1] for r in rows))} tracks, {len(rows)} cajas')
