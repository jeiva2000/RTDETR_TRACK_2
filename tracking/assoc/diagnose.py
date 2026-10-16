"""Diagnóstico del módulo de asociación.

decisions : reparte las decisiones (track, frame) de clips de val -- con las decisiones del propio modelo, como en inferencia --
            en: ok_present | ok_absent | miss (dice 'no aparece' y sí estaba) | swap (elige la detección de OTRA persona GT)
            | wrong_bg (elige una query de fondo) | halluc_person (no estaba y elige la detección de otra persona)
            | halluc_bg (no estaba y elige una query de fondo). Por cruce / no cruce y por frames desde que se vio el track.
per-seq   : HOTA, AssA, IDF1, IDSW por secuencia para varios métodos (TrackEval, una secuencia cada vez).

  python -m tracking.assoc.diagnose decisions --ckpt $OUTPUT_ROOT/assoc_graph/best.pth --cache $FEAT_DIR/val --samples 1000
  python -m tracking.assoc.diagnose per-seq --gt-root $DANCETRACK_DIR/val --methods bytetrack=<dir> ocsort=<dir> graph=<dir>
"""
import argparse, contextlib, io, os, sys
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

CLASSES = ['ok_present', 'ok_absent', 'miss', 'swap', 'wrong_bg', 'halluc_person', 'halluc_bg']


def classify(r):
    present, det = r['tgt'] >= 0, r['pred'] >= 0
    if present and det:
        if r['pred'] == r['tgt']: return 'ok_present'
        return 'swap' if r['pred_gt'] >= 0 else 'wrong_bg'
    if present: return 'miss'
    if not det: return 'ok_absent'
    return 'halluc_person' if r['pred_gt'] >= 0 else 'halluc_bg'


def bucket(g): return '1' if g <= 1 else '2-3' if g <= 3 else '4-10' if g <= 10 else '>10'


def table(rows, title):
    c = Counter(classify(r) for r in rows); n = max(len(rows), 1)
    print(f'{title:<22} n={len(rows):>7} ' + ' '.join(f'{k}={100 * c[k] / n:5.1f}%' for k in CLASSES))


def decisions(a):
    import torch
    from tracking.assoc.model import AssocModel
    from tracking.assoc.train import run_clip
    from tracking.video.clips import CachedClipDataset
    ck = torch.load(a.ckpt, map_location='cpu'); ar = argparse.Namespace(**ck['args'])
    ar.p_fn = ar.p_cross_occ = 0.0  # sin inyección: se mide como en inferencia
    m = AssocModel(ar.update, in_dim=ck['model']['proj.0.weight'].shape[1], dim=ar.dim, hidden=ar.hidden, ema_alpha=ar.ema_alpha,
                   graph_radius=ar.graph_radius, app_aff=ar.app_aff).to(a.device).eval()
    m.load_state_dict(ck['model'])
    ds = CachedClipDataset(a.cache, None, T=ar.T, samples_per_epoch=a.samples, seed=a.seed, max_stride=ar.max_stride, p_skip=ar.p_skip,
                           max_skip=ar.max_skip, gt_root=a.gt_root or ar.gt_root, cross_iou=ar.cross_iou)
    log = []
    with torch.no_grad():
        for i in range(len(ds)):
            run_clip(m, ds[i], ar, a.device, train=False, ss=1.0, log=log)
    print(f'variante={ar.update}  clips={len(ds)}  decisiones={len(log)}  (cruce = par GT con IoU>{ar.cross_iou})')
    table(log, 'TODAS')
    table([r for r in log if r['cross']], 'en cruce')
    table([r for r in log if not r['cross']], 'fuera de cruce')
    for b in ('1', '2-3', '4-10', '>10'):
        table([r for r in log if bucket(r['gap']) == b], f'gap {b} frames')
    sw = [r for r in log if classify(r) == 'swap']
    print(f'swaps: {len(sw)}  ({100 * sum(r["cross"] for r in sw) / max(len(sw), 1):.0f}% en cruce)')


def per_seq(a):
    from tracking.baseline.evaluate import trackeval_run
    from tracking.baseline.mot_io import read_seqinfo
    methods = dict(m.split('=', 1) for m in a.methods)
    seqs = a.seqs or sorted(f[:-4] for f in os.listdir(next(iter(methods.values()))) if f.endswith('.txt'))
    print('seq'.ljust(16) + ''.join(f'{k[:10]:>34}' for k in methods))
    print(' ' * 16 + ''.join(f'{"HOTA":>9}{"AssA":>8}{"IDF1":>9}{"IDSW":>8}' for _ in methods))
    tot = {k: [] for k in methods}
    for s in seqs:
        ln = {s: read_seqinfo(os.path.join(a.gt_root, s))['length']}
        line = s.ljust(16)
        for k, d in methods.items():
            with contextlib.redirect_stdout(io.StringIO()):  # TrackEval imprime ruido por cada llamada
                r = trackeval_run(a.gt_root, d, ln)
            tot[k].append(r)
            line += f'{r["HOTA"]:9.1f}{r["AssA"]:8.1f}{r["IDF1"]:9.1f}{r["IDSW"]:8d}'
        print(line, flush=True)
    print('media'.ljust(16) + ''.join(f'{sum(x["HOTA"] for x in v) / len(v):9.1f}{sum(x["AssA"] for x in v) / len(v):8.1f}'
                                      f'{sum(x["IDF1"] for x in v) / len(v):9.1f}{sum(x["IDSW"] for x in v):8d}' for v in tot.values()))
    print('(media = media simple de las secuencias, no el valor combinado de TrackEval; IDSW = suma)')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('decisions'); d.add_argument('--ckpt', required=True); d.add_argument('--cache', required=True)
    d.add_argument('--samples', type=int, default=1000); d.add_argument('--seed', type=int, default=777)
    d.add_argument('--gt-root', default=None); d.add_argument('--device', default=None, help='por defecto cuda:0 si hay GPU, si no cpu')
    q = sub.add_parser('per-seq'); q.add_argument('--gt-root', default=os.environ.get('DANCETRACK_DIR', '') and os.path.join(os.environ.get('DANCETRACK_DIR', ''), 'val'))
    q.add_argument('--methods', nargs='+', required=True, help='nombre=carpeta_con_<seq>.txt'); q.add_argument('--seqs', nargs='*', default=None)
    a = p.parse_args()
    if a.cmd == 'decisions':
        import torch
        a.device = a.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        decisions(a)
    else:
        per_seq(a)
