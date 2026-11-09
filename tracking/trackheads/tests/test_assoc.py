"""Tests de la métrica de asociación (tracking/trackheads/assoc.py) con casos construidos a mano.
Ejecución: python -m tracking.trackheads.tests.test_assoc   (o pytest tracking/trackheads/tests)"""
import os, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from tracking.trackheads.assoc import assoc_flags, assoc_summary, assoc_sums, gt_cross

XY = torch.tensor([[x, 0.5] for x in (0.1, 0.3, 0.5, 0.7, 0.9)])                    # 5 tokens en una fila


def _case():
    """Imagen 0: el track de la identidad 1 estaba en x=0.40 (prev_box); en t+k la persona 1 está en x=0.30 y la 2 en x=0.50.
    prev_box solapa más con la 2 (falla); la caja predicha va a x=0.30 (acierta); la atención se concentra en el token x=0.3 (acierta).
    Un segundo track (identidad 3) no está detectado en t+k: no cuenta. Un tercero es relleno."""
    tg = dict(track_id=torch.tensor([[1, 3, -1]]), track_mask=torch.tensor([[True, True, False]]), present=torch.tensor([[1., 1., 0.]]),
              prev_box=torch.tensor([[[0.42, 0.5, 0.15, 0.3], [0.8, 0.5, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]]]))
    attn = torch.tensor([[[0.0, 0.9, 0.1, 0.0, 0.0], [0.2] * 5, [0.2] * 5]])
    out = dict(boxes=torch.tensor([[[0.30, 0.5, 0.15, 0.3], [0.8, 0.5, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]]]), attn=attn, token_xy=XY)
    box_k = torch.tensor([[[0.30, 0.5, 0.15, 0.3], [0.50, 0.5, 0.15, 0.3], [0.9, 0.5, 0.05, 0.05], [0.7, 0.5, 0.1, 0.1]]])
    sc_k = torch.tensor([[0.9, 0.9, 0.9, 0.1]])                                       # la última no es candidata (score bajo)
    tid_k = torch.tensor([[1, 2, -1, 3]])                                             # la identidad 3 solo tiene una detección de score bajo
    gt_t = [torch.tensor([[1, .40, .5, .15, .3], [2, .45, .5, .15, .3], [3, .8, .5, .1, .1]])]
    gt_tk = [torch.tensor([[1, .30, .5, .15, .3], [2, .50, .5, .15, .3], [3, .8, .5, .1, .1]])]
    return out, tg, box_k, sc_k, tid_k, gt_t, gt_tk


def test_flags():
    out, tg, box_k, sc_k, tid_k, gt_t, gt_tk = _case()
    f = assoc_flags(out, tg, box_k, sc_k, tid_k, gt_t, gt_tk, det_thr=0.3, cross_iou=0.3)
    assert f['valid'].tolist() == [[True, False, False]]
    assert f['ok_pred'].tolist() == [[True, False, False]]
    assert f['ok_prev'].tolist() == [[False, False, False]]
    assert f['ok_attn'].tolist() == [[True, False, False]]
    assert f['cross'].tolist() == [[True, False, False]]                              # 1 y 2 solapan en t
    s = assoc_summary(assoc_sums(f))
    assert s['assoc_n'] == 1 and s['assoc_pred'] == 1.0 and s['assoc_prev'] == 0.0 and s['assoc_gain'] == 1.0 and s['cross_n'] == 1


def test_gt_cross():
    gt = [torch.tensor([[1, .2, .5, .2, .2], [2, .25, .5, .2, .2], [3, .8, .5, .1, .1]])]
    far = [torch.tensor([[1, .1, .5, .1, .1], [2, .5, .5, .1, .1], [3, .9, .5, .05, .05]])]
    assert gt_cross(torch.tensor([[1, 2, 3, 7]]), gt, far, 0.3).tolist() == [[True, True, False, False]]
    assert gt_cross(torch.tensor([[1, 2, 3]]), far, far, 0.3).tolist() == [[False, False, False]]


def test_no_candidates_and_empty():
    out, tg, box_k, sc_k, tid_k, gt_t, gt_tk = _case()
    f = assoc_flags(out, tg, box_k, torch.zeros_like(sc_k), tid_k, gt_t, gt_tk)
    assert not f['valid'].any() and not f['ok_pred'].any()
    s = assoc_summary(assoc_sums(f))
    assert s['assoc_pred'] is None and s['assoc_gain'] is None


if __name__ == '__main__':
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        fn()
        print('ok', name)
    print(f'{len(tests)} tests OK')
