"""Evaluación de la Fase 7: DetA del detector (a un umbral de score fijo) y precisión de asignación del módulo en clips de val,
ambas con el detector ACTUAL (sin caché)."""
import contextlib, io, os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from tracking.baseline.mot_io import read_mot, read_seqinfo, write_mot
from tracking.video.extractor import topk_queries, match_queries
from tracking.assoc.train import run_clip


def make_strided_gt(gt_root, seqs, stride, out_root):
    """GT con solo los frames (f-1) % stride == 0, renumerados 1..n (para evaluar DetA sobre val con stride). Devuelve {seq: n}."""
    lengths = {}
    for s in seqs:
        gt = read_mot(os.path.join(gt_root, s, 'gt', 'gt.txt'), full=True)
        n = read_seqinfo(os.path.join(gt_root, s))['length']
        sel = gt[(gt[:, 0] - 1) % stride == 0].copy()
        sel[:, 0] = (sel[:, 0] - 1) // stride + 1
        os.makedirs(os.path.join(out_root, s, 'gt'), exist_ok=True)
        np.savetxt(os.path.join(out_root, s, 'gt', 'gt.txt'), sel, delimiter=',', fmt=['%d', '%d'] + ['%.2f'] * 4 + ['%g'] * (sel.shape[1] - 6))
        lengths[s] = (n - 1) // stride + 1
    return lengths


class _Frames(torch.utils.data.Dataset):
    def __init__(self, seq_dir, ext, frames):
        self.dir, self.ext, self.frames = os.path.join(seq_dir, 'img1'), ext, frames
        self.tf = T.Compose([T.Resize((640, 640)), T.ToTensor()])

    def __len__(self): return len(self.frames)

    def __getitem__(self, i):
        return self.tf(Image.open(os.path.join(self.dir, f'{self.frames[i]:08d}{self.ext}')).convert('RGB')), i


@torch.no_grad()
def eval_deta(fwd, val_root, seqs, stride, thr, work_dir, batch=16, num_workers=4, device='cuda:0', extra_thrs=()):
    """DetA (TrackEval) de las detecciones con score >= thr en los frames (f-1) % stride == 0. fwd(images) -> logits, boxes (cxcywh norm).
    Devuelve DetA (a thr), n_det y, si se piden extra_thrs, DetA_by_thr {umbral: DetA} (útil para separar un cambio de calibración
    de scores de un empeoramiento real)."""
    from tracking.baseline.evaluate import trackeval_run
    thrs = [thr] + [t for t in extra_thrs if t != thr]
    gt_dir = os.path.join(work_dir, 'strided_gt')
    lengths = make_strided_gt(val_root, seqs, stride, gt_dir)
    n_det = {t: 0 for t in thrs}
    for s in seqs:
        info = read_seqinfo(os.path.join(val_root, s))
        frames = list(range(1, info['length'] + 1, stride))
        dl = torch.utils.data.DataLoader(_Frames(os.path.join(val_root, s), info['ext'], frames), batch_size=batch, num_workers=num_workers)
        rows = []
        for x, ix in dl:
            logits, boxes = fwd(x.to(device))
            sc = logits.float().sigmoid().max(-1).values.cpu(); bx = boxes.float().cpu()
            for b in range(len(ix)):
                for j in (sc[b] >= min(thrs)).nonzero().flatten().tolist():
                    cx, cy, w, h = bx[b, j].tolist()
                    rows.append((int(ix[b]) + 1, len(rows) + 1, (cx - w / 2) * info['width'], (cy - h / 2) * info['height'],
                                 w * info['width'], h * info['height'], float(sc[b, j])))
        for t in thrs:
            sel = [r for r in rows if r[6] >= t]
            n_det[t] += len(sel)
            write_mot(os.path.join(work_dir, f'strided_dets_{t}', f'{s}.txt'), sel)
    by_thr = {}
    for t in thrs:
        with contextlib.redirect_stdout(io.StringIO()):
            by_thr[t] = trackeval_run(gt_dir, os.path.join(work_dir, f'strided_dets_{t}'), lengths)['DetA']
    out = dict(DetA=by_thr[thr], n_det=n_det[thr])
    if extra_thrs:
        out['DetA_by_thr'] = {str(t): by_thr[t] for t in thrs}
        out['n_det_by_thr'] = {str(t): n_det[t] for t in thrs}
    return out


@torch.no_grad()
def eval_assoc_live(extract, matcher, assoc, ds, ar, device, top_k, min_iou):
    """Precisión de asignación (decisiones del propio modelo, ss=1) en clips de val con el detector actual.
    extract(images (T,3,H,W)) -> logits (T,Q,C), boxes (T,Q,4), emb (T,Q,D)."""
    assoc.eval()
    tot = dict(loss=0.0, n=0, w=0.0, ok=0, ok_present=0, present=0)
    for i in range(len(ds)):
        c = ds[i]
        logits, boxes, emb = extract(c['images'].to(device))
        q = topk_queries(logits, boxes, emb, top_k)
        tid, _ = match_queries(matcher, q, c['gts'], device, min_iou)
        clip = dict(emb=q['emb'].float(), boxes=q['boxes'].float(), scores=q['scores'].float(), track_id=tid, frames=c['frames'])
        l, s = run_clip(assoc, clip, ar, device, train=False, ss=1.0)
        tot['loss'] += float(l)
        for k in ('n', 'w', 'ok', 'ok_present', 'present'):
            tot[k] += s[k]
    assoc.train()
    return dict(loss=tot['loss'] / max(tot['w'], 1), acc=tot['ok'] / max(tot['n'], 1), acc_present=tot['ok_present'] / max(tot['present'], 1))
