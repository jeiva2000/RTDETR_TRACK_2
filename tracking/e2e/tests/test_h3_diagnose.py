"""Tests de tracking/e2e/h3/diagnose.py (diagnóstico del H3): coherencia de los contadores (las categorías suman el total), que cada modo hace lo que dice (teacher no genera cambios de
ID), que un fallo forzado (umbral de probabilidad casi imposible) se detecta y se atribuye a la causa correcta, el resumen de logs y la CLI. CPU, datos sintéticos.
Ejecución: python -m tracking.e2e.tests.test_h3_diagnose"""
import argparse, json, os, subprocess, sys, tempfile
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
sys.path.insert(0, ROOT)
from tracking.e2e.config import deep_update, load_config
from tracking.e2e.h3 import diagnose as D
from tracking.e2e.model import build_model
from tracking.e2e.tests.test_h3 import _synthetic

H3_UPDATE = ['h3.enabled=true', 'h3.dim=64', 'h3.heads=4', 'h3.ffn=128', 'h3.gru_dim=32', 'h3.app_dim=32', 'h3.K_hist=6', 'state.max_tracks=8',
             'train.epochs=2', 'train.samples_per_epoch=8', 'train.batch=2', 'train.T=6', 'train.val_samples=2']


def _trained(tmp):
    from tracking.e2e.train import main
    _synthetic(tmp)
    main(argparse.Namespace(cfg=None, seed=0, device='cpu', gt_root=None, out=f'{tmp}/run', train_cache=f'{tmp}/train/cache', val_cache=f'{tmp}/val/cache', update=H3_UPDATE))
    ck = torch.load(f'{tmp}/run/best.pth')
    cfg = load_config(None, ck['cfg'])
    model = build_model(cfg, ck['in_dim']).eval(); model.load_state_dict(ck['model'])
    return model, cfg, ck


def _caches(tmp):
    return {'va00': torch.load(f'{tmp}/val/cache/va00.pt')}


def _check_invariants(r):
    c = r['counts']
    assert c['present'] == c['correct'] + c['missed'] + c['swap_to_other_id'] + c['swap_to_fp'], 'las decisiones con detección presente suman el total'
    assert c['absent'] == c['absent_ok'] + c['false_link'] and c['false_link'] == c['false_link_to_fp'] + c['false_link_to_other_id']
    assert all(v is None or 0.0 <= v <= 1.0 for v in r['top1_present'].values())
    kept = [t['true_pair_kept'] for t in r['threshold_curve'] if t['true_pair_kept'] is not None]
    assert all(a >= b - 1e-9 for a, b in zip(kept, kept[1:])), 'subir el umbral nunca conserva más parejas verdaderas'
    absent = [t['absent_track_linked'] for t in r['threshold_curve'] if t['absent_track_linked'] is not None]
    assert all(a >= b - 1e-9 for a, b in zip(absent, absent[1:]))
    if r['mode'] == 'closed':
        ic = r['id_changes']
        assert ic['total'] == sum(ic['by_cause'].values()) == sum(c[k] for k in c if k.startswith('idchg_'))
        assert r['ids']['predicted'] == sum(c.get(k, 0) for k in ('births_new_identity', 'births_rebirth', 'births_fp')), 'cada ID creado es un nacimiento clasificado'
    else:
        assert 'id_changes' not in r and 'births' not in r, 'teacher no mide cambios de ID: la memoria usa los IDs GT'


def test_report_invariants_and_modes():
    with tempfile.TemporaryDirectory() as tmp:
        model, cfg, _ = _trained(tmp)
        rep = D.diagnose(model, cfg, _caches(tmp), 'cpu', ['closed', 'teacher'])
        for mode in ('closed', 'teacher'):
            _check_invariants(rep[mode])
            assert rep[mode]['decisions_present']['n'] > 0 and 'motion' in rep[mode] and rep[mode]['gates_mean_all_pairs']
        assert 'ids' in rep['closed'] and rep['closed']['ids']['gt'] == 4, 'la secuencia sintética tiene 4 identidades'
        md = D.to_markdown(rep, dict(ckpt='x', cache='y', n_seqs=1, min_prob=0.5, new_thr=0.6, det_thr=0.3, max_age=30))
        assert '## Modo closed' in md and '## Modo teacher' in md and '## Pistas' in md
        assert isinstance(D.hints(rep), list) and D.hints(rep)
        json.dumps(rep)                                                    # serializable (sin tipos de numpy)


def test_forced_failure_is_detected_and_attributed():
    """Con h3.min_prob casi 1 ninguna pareja se acepta: los tracks quedan 'rechazados' y cada detección nace como ID nuevo con el track anterior aún vivo."""
    with tempfile.TemporaryDirectory() as tmp:
        model, cfg, _ = _trained(tmp)
        good = D.diagnose(model, cfg, _caches(tmp), 'cpu', ['closed'])['closed']
        deep_update(cfg, {'h3': {'min_prob': 0.9999}})
        model.hc = cfg.h3
        bad = D.diagnose(model, cfg, _caches(tmp), 'cpu', ['closed'])['closed']
        _check_invariants(bad)
        assert good['decisions_present']['missed'] < 0.05 and bad['decisions_present']['missed'] > 0.9
        assert good['ids']['ratio'] < 1.5 and bad['ids']['ratio'] > 2 * good['ids']['ratio'], 'el umbral imposible fragmenta las identidades (el normal puede crear algún ID extra tras una oclusión larga)'
        assert bad['id_changes']['by_cause'].get('rebirth_prev_alive', 0) == max(bad['id_changes']['by_cause'].values()), 'la causa principal es el duplicado con el track anterior vivo'
        assert bad['decisions_present']['missed_but_top1_s_was_right'] > 0.9, 'la afinidad sí tenía la detección correcta: es el umbral, no la señal'
        assert any('umbral' in h for h in D.hints(dict(closed=bad)))


def test_summarize_logs():
    with tempfile.TemporaryDirectory() as tmp:
        for name, recs in (('e2e_unfrozen_h1', [dict(epoch=0, val_acc=0.5, val_acc_present=0.6, val_loss=1.0, deta_ok=True), dict(epoch=1, val_acc=0.9, val_acc_present=0.95, val_loss=0.4, deta_ok=True),
                                                 dict(epoch=2, val_acc=0.99, val_acc_present=0.99, val_loss=0.1, deta_ok=False)]),
                           ('e2e_unfrozen_h3', [dict(epoch=0, val_acc=0.7, val_loss=0.5)])):                      # sin deta_ok (train.py): se toma por válido
            os.makedirs(f'{tmp}/{name}')
            open(f'{tmp}/{name}/log.txt', 'w').write('\n'.join(json.dumps(r) for r in recs) + '\n')
        t = D.summarize_logs([f'{tmp}/e2e_unfrozen_h1', f'{tmp}/e2e_unfrozen_h2', f'{tmp}/e2e_unfrozen_h3'])
        rows = {l.split('|')[1].strip(): l for l in t.splitlines()[2:]}
        assert '| 3 | sí (DetA) | 1 | 0.9000 |' in rows['h1'], 'la mejor época válida excluye la que cayó por DetA (0.99)'
        assert '(sin log.txt)' in rows['h2'] and '| 1 | no | 0 | 0.7000 |' in rows['h3']


def test_cli_writes_report():
    with tempfile.TemporaryDirectory() as tmp:
        _, _, ck = _trained(tmp)
        cmd = [sys.executable, '-W', 'ignore', '-m', 'tracking.e2e.h3.diagnose', 'run', '--ckpt', f'{tmp}/run/best.pth', '--cache', f'{tmp}/val/cache', '--out', f'{tmp}/diag',
               '--modes', 'teacher', '--max-seqs', '1', '-u', 'h3.min_prob=0.3']
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stderr[-1500:]
        rep = json.load(open(f'{tmp}/diag/report.json'))
        assert rep['meta']['min_prob'] == 0.3 and 'teacher' in rep and 'closed' not in rep and open(f'{tmp}/diag/report.md').read().startswith('# Diagnóstico del H3')
        r2 = subprocess.run([sys.executable, '-W', 'ignore', '-m', 'tracking.e2e.h3.diagnose', 'logs', '--runs', f'{tmp}/nada'], cwd=ROOT, capture_output=True, text=True)
        assert r2.returncode == 0 and '(sin log.txt)' in r2.stdout


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for t in tests:
        t(); print('ok  ', t.__name__)
    print(f'{len(tests)} tests correctos')
