"""Tests del diagnóstico de cambios de identidad (tracking/trackheads/idsw.py) con secuencias construidas a mano.
Ejecución: python -m tracking.trackheads.tests.test_idsw   (o pytest tracking/trackheads/tests)"""
import os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from tracking.trackheads.idsw import sequence_events, summarize


def rows(spec):
    """spec: lista de (frame, id, x) con cajas 10×20 en y=0 -> array MOT."""
    return np.array([[f, i, x, 0, 10, 20, 1, 1, 1] for f, i, x in spec], dtype=float)


def test_rebirth_swap_and_back():
    # GT: A (x=0) en frames 1-20, B (x=100) en frames 1-20 (lejos: sin cruce)
    gt = rows([(f, 1, 0) for f in range(1, 21)] + [(f, 2, 100) for f in range(1, 21)])
    pr = []
    # A: id 10 en 1-5; hueco 6-8 (sin predicción); renace como id 11 en 9-20, y 10 no vuelve
    pr += [(f, 10, 0) for f in range(1, 6)] + [(f, 11, 0) for f in range(9, 21)]
    # B: id 20 en 1-14; id 21 nace en 15-16 (sin hueco); id 20 vuelve en 17-20
    pr += [(f, 20, 100) for f in range(1, 15)] + [(f, 21, 100) for f in (15, 16)] + [(f, 20, 100) for f in range(17, 21)]
    ev = sorted(sequence_events(gt, rows(pr)), key=lambda e: (e['gt'], e['frame']))
    assert [(e['gt'], e['frame'], e['q'], e['p'], e['gap'], e['new']) for e in ev] == \
        [(1, 9, 10, 11, 3, True), (2, 15, 20, 21, 0, True), (2, 17, 21, 20, 0, False)]
    assert not ev[0]['q_alive'] and not ev[0]['back'] and ev[1]['q_alive'] and ev[1]['back']
    assert not any(e['cross'] or e['swap'] for e in ev)
    s = summarize(ev)
    assert s['idsw'] == 3 and s['renacimiento_%'] == 66.7 and s['  hueco 1-10'] == 33.3 and s['  hueco 0'] == 33.3


def test_swap_in_crossing():
    # A y B se cruzan (x=0 y x=4, solapan) y los ids 10 y 20 se intercambian en el frame 6
    gt = rows([(f, 1, 0) for f in range(1, 11)] + [(f, 2, 4) for f in range(1, 11)])
    pr = rows([(f, 10, 0 if f < 6 else 4) for f in range(1, 11)] + [(f, 20, 4 if f < 6 else 0) for f in range(1, 11)])
    ev = sequence_events(gt, pr)
    assert len(ev) == 2 and all(e['swap'] and not e['new'] and e['cross'] and e['gap'] == 0 for e in ev)
    assert summarize(ev)['  intercambio (swap)'] == 100.0


if __name__ == '__main__':
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        fn()
        print('ok', name)
    print(f'{len(tests)} tests OK')
