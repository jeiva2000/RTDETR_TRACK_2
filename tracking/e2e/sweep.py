"""Barrido de los umbrales de inferencia del módulo e2e y selección de variante (misma regla para todas: HOTA medio sobre las semillas, en val).

  python -m tracking.e2e.sweep run --runs <run_s1> <run_s2> ... --cache <FEAT_DIR/val> --gt-root <DANCETRACK_DIR/val> --out <dir>
      Prueba cada configuración de la rejilla (birth_mode head|score, birth_thr, min_score) en cada semilla. Escribe <out>/sweep.json y
      <out>/best.json (mejor por HOTA medio) y, con esa configuración, <run>/eval.json (global y cruces) y <run>/best_infer.json en cada run,
      de modo que aggregate_seeds.sh recoja directamente el resultado con el umbral elegido.
  python -m tracking.e2e.sweep select --variants nombre=<out_dir> ... --out best_variant.json
      Elige la variante con mayor HOTA medio a su mejor configuración.
Aviso: el umbral se elige sobre val y se reporta sobre val (optimista); debe hacerse igual para todos los métodos comparados."""
import argparse, itertools, json, os, shutil, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tracking.baseline.evaluate import evaluate, trackeval_run
from tracking.baseline.mot_io import read_seqinfo, write_mot
from tracking.e2e.config import deep_update, load_config, parse_overrides
from tracking.e2e.infer import track_sequence
from tracking.e2e.model import build_model

KEYS = ('HOTA', 'DetA', 'AssA', 'IDF1', 'IDSW')


def make_grid(thrs, scores, rule):
    g = [dict(birth_mode='head', birth_thr=t, min_score=s) for t, s in itertools.product(thrs, scores)]
    if rule:
        g += [dict(birth_mode='score', birth_thr=0.0, min_score=s) for s in scores]
    return g


def make_grid_h3(probs, scores, guards=(0.0,), dedupes=(0.0,), det_thrs=(None,)):
    """Rejilla del H3 (no hay cabeza de nacimiento): birth_mode='h3', birth_thr = h3.min_prob (umbral de probabilidad de la pareja), min_score = h3.new_thr, y los mecanismos contra
    duplicados (h3/loop_breakers.py): guard = h3.birth_guard_iou, dedupe = h3.dedupe_iou (0 = apagado; solo con guarda) y det_thr = h3.det_thr (None = el entrenado)."""
    return [dict(birth_mode='h3', birth_thr=p, min_score=s, guard=g, dedupe=d, det_thr=dt) for p, s, g, d, dt in itertools.product(probs, scores, guards, dedupes, det_thrs)
            if g or not d]   # dedupe sin guarda NO se prueba: el duplicado eliminado vuelve a nacer en el frame siguiente (medido en sintético: IDs x4 -> x11)


def cfg_id(c):
    return (f"{c['birth_mode']}_thr{c['birth_thr']}_s{c['min_score']}" + (f"_g{c['guard']}" if c.get('guard') else '') + (f"_d{c['dedupe']}" if c.get('dedupe') else '') +
            (f"_dt{c['det_thr']}" if c.get('det_thr') is not None else ''))


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location='cpu')
    cfg = load_config(None, ck['cfg'])
    model = build_model(cfg, ck['in_dim']).to(device).eval()
    model.load_state_dict(ck['model'])
    return model, cfg


def run_inference(model, cfg, caches, out_dir, c, device):
    if cfg.h3.enabled:
        ov = [f"h3.min_prob={c['birth_thr']}", f"h3.new_thr={c['min_score']}", f"h3.birth_guard_iou={c.get('guard') or 'null'}", f"h3.dedupe_iou={c.get('dedupe') or 'null'}"]
        trained = cfg.__dict__.setdefault('_det_thr_trained', cfg.h3.det_thr)                  # el barrido muta cfg: se recuerda el det_thr entrenado para no arrastrar el de otra configuración
        ov.append(f"h3.det_thr={trained if c.get('det_thr') is None else c['det_thr']}")
        deep_update(cfg, parse_overrides(ov))
    else:
        deep_update(cfg, parse_overrides([f"infer.birth_mode={c['birth_mode']}", f"infer.birth_thr={c['birth_thr']}", f"infer.min_score={c['min_score']}"]))
    os.makedirs(out_dir, exist_ok=True)
    for s, d in caches.items():
        rows, _ = track_sequence(model, d, cfg, device)
        write_mot(os.path.join(out_dir, f'{s}.txt'), rows)


def sweep(a):
    seqs = a.seqs or sorted(f[:-3] for f in os.listdir(a.cache) if f.endswith('.pt'))
    caches = {s: torch.load(os.path.join(a.cache, f'{s}.pt')) for s in seqs}
    lengths = {s: read_seqinfo(os.path.join(a.gt_root, s))['length'] for s in seqs}
    use = a.runs[:a.sweep_seeds] if a.sweep_seeds else a.runs
    models = {r: load_model(os.path.join(r, a.ckpt_name), a.device) for r in a.runs}
    h3 = models[a.runs[0]][1].h3.enabled
    grid = make_grid_h3(a.min_prob, a.score, a.guard_iou, a.dedupe_iou, a.det_thr or [None]) if h3 else make_grid(a.thr, a.score, not a.no_rule)
    os.makedirs(a.out, exist_ok=True)
    results = []
    for c in grid:
        per = []
        for r in use:
            m, cfg = models[r]
            tmp = os.path.join(a.out, 'tmp', f'{cfg_id(c)}_{os.path.basename(r.rstrip("/"))}')
            run_inference(m, cfg, caches, tmp, c, a.device)
            per.append(trackeval_run(a.gt_root, tmp, lengths))
            shutil.rmtree(tmp, ignore_errors=True)
        mean = {k: float(np.mean([p[k] for p in per])) for k in KEYS}
        std = {k: float(np.std([p[k] for p in per], ddof=1)) if len(per) > 1 else 0.0 for k in KEYS}
        results.append(dict(cfg=c, mean=mean, std=std, per_seed=per))
        print(f"{cfg_id(c):28} HOTA {mean['HOTA']:5.1f}±{std['HOTA']:3.1f} DetA {mean['DetA']:5.1f} AssA {mean['AssA']:5.1f} IDF1 {mean['IDF1']:5.1f} IDSW {mean['IDSW']:6.0f}", flush=True)
    best = max(results, key=lambda x: x['mean']['HOTA'])
    json.dump(results, open(os.path.join(a.out, 'sweep.json'), 'w'), indent=1)
    json.dump(dict(best=best['cfg'], mean=best['mean'], std=best['std'], seeds=len(use), runs=a.runs), open(os.path.join(a.out, 'best.json'), 'w'), indent=1)
    print('MEJOR:', cfg_id(best['cfg']), best['mean'], flush=True)
    for r in a.runs:   # evaluación completa (global y cruces) con la mejor configuración, en todas las semillas
        m, cfg = models[r]
        res = os.path.join(r, 'results_best')
        run_inference(m, cfg, caches, res, best['cfg'], a.device)
        json.dump(evaluate(res, a.gt_root, seqs, margin=a.margin, iou_thr=a.cross_iou), open(os.path.join(r, 'eval.json'), 'w'), indent=1)
        json.dump(best['cfg'], open(os.path.join(r, 'best_infer.json'), 'w'))
    shutil.rmtree(os.path.join(a.out, 'tmp'), ignore_errors=True)


def select(a):
    rows = []
    for item in a.variants:
        name, d = item.split('=', 1)
        b = json.load(open(os.path.join(d, 'best.json')))
        rows.append(dict(variant=name, hota=b['mean']['HOTA'], std=b['std']['HOTA'], cfg=b['best'], mean=b['mean'], dir=d))
    rows.sort(key=lambda r: -r['hota'])
    for r in rows:
        print(f"{r['variant']:24} HOTA {r['hota']:5.1f}±{r['std']:3.1f}  {cfg_id(r['cfg'])}")
    json.dump(dict(variant=rows[0]['variant'], best_infer=rows[0]['cfg'], ranking=rows), open(a.out, 'w'), indent=1)
    print('variante elegida:', rows[0]['variant'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--runs', nargs='+', required=True, help='carpetas de run (una por semilla) con el checkpoint')
    r.add_argument('--ckpt-name', default='best.pth', help='best.pth (etapa congelada) o e2e_best.pth (etapa descongelada)')
    r.add_argument('--cache', required=True); r.add_argument('--gt-root', required=True); r.add_argument('--out', required=True)
    r.add_argument('--seqs', nargs='*', default=None)
    r.add_argument('--thr', type=float, nargs='+', default=[0.05, 0.1, 0.3, 0.5]); r.add_argument('--score', type=float, nargs='+', default=[0.5, 0.6])
    r.add_argument('--min-prob', type=float, nargs='+', default=[0.5], help='solo H3: umbral de probabilidad de la pareja (h3.min_prob); por defecto solo 0.5, pasar varios para barrerlo')
    r.add_argument('--guard-iou', type=float, nargs='+', default=[0.0], help='solo H3: h3.birth_guard_iou (no nace una detección cubierta por un track vivo con IoU >= valor); 0 = apagado')
    r.add_argument('--dedupe-iou', type=float, nargs='+', default=[0.0], help='solo H3: h3.dedupe_iou (se elimina el track más joven de dos solapados con IoU >= valor); 0 = apagado')
    r.add_argument('--det-thr', type=float, nargs='+', default=None, help='solo H3: h3.det_thr de la inferencia (por defecto el del entrenamiento)')
    r.add_argument('--no-rule', action='store_true', help='no probar el nacimiento por regla (solo cabeza)')
    r.add_argument('--sweep-seeds', type=int, default=0, help='usar solo las primeras N semillas para elegir (0 = todas)')
    r.add_argument('--margin', type=int, default=10); r.add_argument('--cross-iou', type=float, default=0.3)
    r.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    s = sub.add_parser('select'); s.add_argument('--variants', nargs='+', required=True); s.add_argument('--out', required=True)
    a = p.parse_args()
    sweep(a) if a.cmd == 'run' else select(a)
