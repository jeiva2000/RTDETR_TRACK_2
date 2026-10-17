"""Tabla comparativa (markdown) a partir de los JSON de evaluate.py.
Uso: python -m tracking.baseline.make_table nombre=ruta.json nombre2=ruta_s1.json,ruta_s2.json,ruta_s3.json ...
Si una entrada trae varias rutas separadas por comas (p. ej. una por semilla), se muestra media ± desviación típica (n semillas)."""
import json, sys
import numpy as np

KEYS = ('HOTA', 'DetA', 'AssA', 'IDF1', 'IDSW')


def fmt(vals, k):
    v = np.array([d[k] for d in vals], dtype=float)
    if len(v) == 1:
        return f'{int(v[0])}' if k == 'IDSW' else f'{v[0]:.1f}'
    sd = v.std(ddof=1)
    return f'{v.mean():.0f} ± {sd:.0f}' if k == 'IDSW' else f'{v.mean():.1f} ± {sd:.1f}'


def row(name, vals, scope):
    n = f' (n={len(vals)})' if len(vals) > 1 else ''
    return f'| {name}{n} | {scope} | ' + ' | '.join(fmt(vals, k) for k in KEYS) + ' |'


if __name__ == '__main__':
    print('| Método | Subconjunto | HOTA | DetA | AssA | IDF1 | IDSW |\n|---|---|---|---|---|---|---|')
    extra = []
    for arg in sys.argv[1:]:
        name, paths = arg.split('=', 1)
        rs = [json.load(open(p)) for p in paths.split(',')]
        print(row(name, [r['global_'] for r in rs], 'global'))
        cs = [r['crossings'] for r in rs if 'HOTA' in r['crossings']]
        if cs:
            print(row(name, cs, f'cruces ({cs[0]["segments"]} tramos, {cs[0]["frames"]} frames)'))
            for thr in cs[0]['merged']['by_iou']:
                m = np.mean([c['merged']['by_iou'][thr]['merged'] for c in cs])
                ref = cs[0]['merged_gt_reference']['by_iou'][thr]['merged']
                tot = np.mean([c['merged']['by_iou'][thr]['preds'] for c in cs])
                extra.append(f'| {name} | {thr} | {m:.0f} | {ref} | {tot:.0f} |')
    if extra:
        print('\nPredicciones que cubren a >= 2 personas GT en los cruces (IoU >= umbral con ambas)\n')
        print('| Método | IoU | Predicciones | Referencia (GT como predicción) | Total predicciones |\n|---|---|---|---|---|')
        print('\n'.join(extra))
