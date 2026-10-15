"""Tabla comparativa (markdown) a partir de los JSON de evaluate.py.  Uso: python -m tracking.baseline.make_table nombre=ruta.json ..."""
import json, sys


def row(name, d, scope):
    return f'| {name} | {scope} | {d["HOTA"]:.1f} | {d["DetA"]:.1f} | {d["AssA"]:.1f} | {d["IDF1"]:.1f} | {d["IDSW"]} |'


if __name__ == '__main__':
    print('| Método | Subconjunto | HOTA | DetA | AssA | IDF1 | IDSW |\n|---|---|---|---|---|---|---|')
    extra = []
    for arg in sys.argv[1:]:
        name, path = arg.split('=', 1)
        r = json.load(open(path))
        print(row(name, r['global_'], 'global'))
        c = r['crossings']
        if 'HOTA' in c:
            print(row(name, c, f'cruces ({c["segments"]} tramos, {c["frames"]} frames)'))
            for thr, v in c['merged']['by_iou'].items():
                ref = c['merged_gt_reference']['by_iou'][thr]
                extra.append(f'| {name} | {thr} | {v["merged"]} | {ref["merged"]} | {v["preds"]} |')
    if extra:
        print('\nPredicciones que cubren a >= 2 personas GT en los cruces (IoU >= umbral con ambas)\n')
        print('| Método | IoU | Predicciones | Referencia (GT como predicción) | Total predicciones |\n|---|---|---|---|---|')
        print('\n'.join(extra))
