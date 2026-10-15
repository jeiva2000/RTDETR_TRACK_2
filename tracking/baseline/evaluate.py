"""Evalúa resultados de tracking con TrackEval (HOTA, DetA, AssA, IDF1, IDSW, MOTA): global y en el subconjunto de cruces.

Uso:
  python -m tracking.baseline.evaluate --results <dir con <seq>.txt> --gt-root $DANCETRACK_DIR/val --out <json> [--seqs ...]
"""
import argparse, json, os, shutil, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tracking.baseline.mot_io import read_seqinfo
from tracking.baseline import crossings as C


def trackeval_run(gt_root, results_dir, seq_lengths):
    from tracking.third_party import trackeval
    work = tempfile.mkdtemp(prefix='te_')
    name = 'tracker'
    os.makedirs(os.path.join(work, name))
    for s in seq_lengths:  # TrackEval espera TRACKERS_FOLDER/<name>/<seq>.txt
        src = os.path.join(results_dir, f'{s}.txt')
        dst = os.path.join(work, name, f'{s}.txt')
        if os.path.isfile(src): shutil.copy(src, dst)
        else: open(dst, 'w').close()
    ec = dict(USE_PARALLEL=False, PRINT_RESULTS=False, PRINT_CONFIG=False, TIME_PROGRESS=False,
              OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False, PLOT_CURVES=False)
    dc = dict(GT_FOLDER=gt_root, TRACKERS_FOLDER=work, OUTPUT_FOLDER=work, TRACKERS_TO_EVAL=[name],
              BENCHMARK='DanceTrack', SPLIT_TO_EVAL='val', SKIP_SPLIT_FOL=True, TRACKER_SUB_FOLDER='',
              SEQ_INFO=dict(seq_lengths), PRINT_CONFIG=False)
    ds = trackeval.datasets.MotChallenge2DBox(dc)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(dict(PRINT_CONFIG=False)), trackeval.metrics.Identity(dict(PRINT_CONFIG=False))]
    out, _ = trackeval.Evaluator(ec).evaluate([ds], metrics)
    r = out['MotChallenge2DBox'][name]['COMBINED_SEQ']['pedestrian']
    shutil.rmtree(work, ignore_errors=True)
    return dict(HOTA=100 * float(np.mean(r['HOTA']['HOTA'])), DetA=100 * float(np.mean(r['HOTA']['DetA'])),
                AssA=100 * float(np.mean(r['HOTA']['AssA'])), IDF1=100 * float(r['Identity']['IDF1']),
                IDSW=int(r['CLEAR']['IDSW']), MOTA=100 * float(r['CLEAR']['MOTA']))


def evaluate(results_dir, gt_root, seqs=None, iou_thr=0.3, margin=10, merged_thrs=(0.3, 0.5), workdir=None):
    seqs = seqs or sorted(d for d in os.listdir(gt_root) if os.path.isfile(os.path.join(gt_root, d, 'seqinfo.ini'))
                          and os.path.isfile(os.path.join(results_dir, f'{d}.txt')))
    assert seqs, 'no hay secuencias con GT y resultados'
    lengths = {s: read_seqinfo(os.path.join(gt_root, s))['length'] for s in seqs}
    out = dict(seqs=seqs, global_=trackeval_run(gt_root, results_dir, lengths))
    tmp = workdir or tempfile.mkdtemp(prefix='cross_')
    sub_gt, sub_res = os.path.join(tmp, 'gt'), os.path.join(tmp, 'res')
    index = C.build_subset(gt_root, seqs, sub_gt, iou_thr, margin, lengths)
    out['crossings'] = dict(iou_thr=iou_thr, margin=margin, segments=len(index),
                            frames=sum(e['length'] for e in index.values()))
    if index:
        C.cut_results(results_dir, index, sub_res)
        out['crossings'].update(trackeval_run(sub_gt, sub_res, {k: e['length'] for k, e in index.items()}))
        out['crossings']['merged'] = C.merged_stats(gt_root, results_dir, index, merged_thrs)
        # referencia: el propio GT como "predicción" (cuenta los casos inevitables, donde dos personas GT ya se solapan >= thr)
        ref = os.path.join(tmp, 'gt_as_pred'); os.makedirs(ref, exist_ok=True)
        for s_ in seqs:
            shutil.copy(os.path.join(gt_root, s_, 'gt', 'gt.txt'), os.path.join(ref, f'{s_}.txt'))
        out['crossings']['merged_gt_reference'] = C.merged_stats(gt_root, ref, index, merged_thrs)
    if not workdir:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results', required=True)
    p.add_argument('--gt-root', default=os.environ.get('DANCETRACK_DIR', '') and os.path.join(os.environ.get('DANCETRACK_DIR', ''), 'val'))
    p.add_argument('--seqs', nargs='*', default=None)
    p.add_argument('--cross-iou', type=float, default=0.3)
    p.add_argument('--margin', type=int, default=10)
    p.add_argument('--merged-iou', type=float, nargs='+', default=[0.3, 0.5])
    p.add_argument('--out', default=None)
    a = p.parse_args()
    r = evaluate(a.results, a.gt_root, a.seqs, a.cross_iou, a.margin, tuple(a.merged_iou))
    print(json.dumps(r, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(r, open(a.out, 'w'), indent=1)
