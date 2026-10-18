"""Tests del H1: shapes, enmascarado con número variable de tracks/detecciones, actualización con/sin match, estado (spawn/kill),
asignación exclusiva, desactivación por config y un test de humo (entrenamiento + inferencia sobre datos sintéticos, CPU).
Ejecución: python -m tracking.e2e.tests.test_h1   (o pytest tracking/e2e/tests)"""
import os, sys, tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from tracking.e2e.config import load_config
from tracking.e2e.losses import IGNORE, assoc_loss
from tracking.e2e.model import E2EModel
from tracking.e2e.state import TrackState
from tracking.e2e.tokens import DetTokens

D = 256


def _model(**ov):
    torch.manual_seed(0)
    return E2EModel(load_config(None, ov or None), D).eval()


def _dets(B, K, seed=1):
    g = torch.Generator().manual_seed(seed)
    return DetTokens(torch.randn(B, K, D, generator=g), torch.rand(B, K, 4, generator=g) * 0.3 + 0.2, torch.rand(B, K, generator=g))


def _state(model, B, N, alive, seed=2):
    g = torch.Generator().manual_seed(seed)
    s = TrackState.empty(B, N, D, 'cpu')
    al = torch.zeros(B, N, dtype=torch.bool)
    for b, n in enumerate(alive):
        al[b, :n] = True
    return s.replace(a=torch.randn(B, N, D, generator=g), box=torch.rand(B, N, 4, generator=g) * 0.3 + 0.2,
                     last=torch.full((B, N), 3), ids=torch.where(al, torch.arange(N)[None].expand(B, N) + 10, -torch.ones(B, N, dtype=torch.long)), alive=al)


def test_shapes():
    m, B, N, K = _model(), 2, 8, 5
    logits, birth, enc = m.scores(_state(m, B, N, [3, 5]), _dets(B, K), torch.tensor([4, 4]))
    assert logits.shape == (B, N, K + 1) and birth.shape == (B, K) and enc.shape == (B, K, D)


def test_padding_invariance():
    """Añadir huecos de track libres y detecciones inválidas (relleno) no cambia la salida de los reales."""
    m = _model()
    small_s, small_d = _state(m, 1, 4, [3]), _dets(1, 4)
    big_s = TrackState.empty(1, 7, D, 'cpu')
    big_s = big_s.replace(a=torch.cat([small_s.a, torch.randn(1, 3, D)], 1), box=torch.cat([small_s.box, torch.rand(1, 3, 4)], 1),
                          last=torch.cat([small_s.last, torch.zeros(1, 3, dtype=torch.long)], 1),
                          ids=torch.cat([small_s.ids, -torch.ones(1, 3, dtype=torch.long)], 1),
                          alive=torch.cat([small_s.alive, torch.zeros(1, 3, dtype=torch.bool)], 1))
    big_d = DetTokens(torch.cat([small_d.emb, torch.randn(1, 2, D)], 1), torch.cat([small_d.box, torch.rand(1, 2, 4)], 1),
                      torch.cat([small_d.score, torch.rand(1, 2)], 1), torch.tensor([[True] * 4 + [False] * 2]))
    f = torch.tensor([4])
    l1, b1, _ = m.scores(small_s, small_d, f)
    l2, b2, _ = m.scores(big_s, big_d, f)
    assert torch.allclose(l1[0, :3, :4], l2[0, :3, :4], atol=1e-5) and torch.allclose(l1[0, :3, -1], l2[0, :3, -1], atol=1e-5)
    assert torch.isinf(l2[0, :3, 4:6]).all(), 'las detecciones de relleno deben quedar enmascaradas (-inf)'
    assert torch.allclose(b1[0], b2[0, :4], atol=1e-5), 'la cabeza de nacimiento no depende del relleno'


def test_update_with_and_without_match():
    m, K = _model(), 5
    s, d = _state(m, 1, 4, [2]), _dets(1, K)
    f = torch.tensor([9])
    _, _, enc = m.scores(s, d, f)
    assign = torch.tensor([[1, K, K, K]])        # track 0 -> detección 1; track 1 -> sin match
    s2, matched = m.update(s, assign, d, enc, f)
    assert matched.tolist() == [[True, False, False, False]]
    assert torch.allclose(s2.a[0, 0], 0.9 * s.a[0, 0] + 0.1 * enc[0, 1], atol=1e-6)
    assert torch.equal(s2.box[0, 0], d.box[0, 1]) and int(s2.last[0, 0]) == 9
    assert torch.equal(s2.a[0, 1], s.a[0, 1]) and torch.equal(s2.box[0, 1], s.box[0, 1]) and int(s2.last[0, 1]) == 3, 'sin match no cambia nada'


def test_spawn_and_kill():
    s = TrackState.empty(2, 3, D, 'cpu')
    s, n = s.spawn(0, torch.tensor([5, 6]), torch.randn(2, D), torch.rand(2, 4), torch.tensor(1))
    assert n == 2 and s.alive[0].tolist() == [True, True, False] and s.ids[0].tolist() == [5, 6, -1] and not s.alive[1].any()
    s, n = s.spawn(0, torch.tensor([7, 8]), torch.randn(2, D), torch.rand(2, 4), torch.tensor(2))
    assert n == 1, 'solo había un hueco libre: el resto se pierde de forma controlada'
    s = s.kill(torch.tensor([100, 100]), max_age=30)
    assert not s.alive.any() and (s.ids == -1).all()


def test_exclusive_assignment():
    m = _model()
    K = 3
    logits = torch.full((1, 2, K + 1), -5.0)
    logits[0, :, 0] = 5.0               # los dos tracks prefieren la detección 0
    logits[0, 0, 0], logits[0, 1, 0] = 6.0, 5.0
    logits[0, :, 1] = 4.0
    assign = m.decide(logits, torch.tensor([[True, True]]))
    assert assign[0, 0] == 0 and assign[0, 1] == 1 and len(set(assign[0].tolist())) == 2


def test_variable_number_of_tracks_in_loss():
    m = _model()
    s = _state(m, 2, 6, [1, 4])
    logits, _, _ = m.scores(s, _dets(2, 5), torch.tensor([4, 4]))
    target = torch.where(s.alive, torch.zeros(2, 6, dtype=torch.long), torch.full((2, 6), IGNORE))
    loss, n = assoc_loss(logits, target)
    assert int(n) == 5 and torch.isfinite(loss)
    leaf = logits.detach().clone().requires_grad_(True)
    assoc_loss(leaf, target)[0].backward()
    assert leaf.grad[0, 1:].abs().sum() == 0 and leaf.grad[1, 4:].abs().sum() == 0, 'los huecos de relleno no reciben gradiente'


def test_loss_finite_without_decisions_and_with_masked_detections():
    """Regresión: sin tracks (0 decisiones) y con detecciones enmascaradas (-inf) la pérdida debe ser 0 y finita, no NaN."""
    m = _model()
    s = _state(m, 2, 6, [0, 0])
    d = _dets(2, 5)
    d = d.drop(torch.tensor([[True, False, False, False, False], [False, True, True, False, False]]))
    logits, birth, _ = m.scores(s, d, torch.tensor([4, 4]))
    assert torch.isneginf(logits).any(), 'el escenario debe incluir columnas enmascaradas'
    loss, n = assoc_loss(logits, torch.full((2, 6), IGNORE))
    assert int(n) == 0 and torch.isfinite(loss) and float(loss) == 0.0
    assert torch.isfinite(birth).all()


def test_disabled_by_config():
    from tracking.e2e.train import main
    import argparse
    a = argparse.Namespace(cfg=None, update=['e2e.enabled=false'], train_cache='x', val_cache='x', out='x', gt_root=None, seed=0, device='cpu')
    try:
        main(a)
    except AssertionError as e:
        assert 'desactivado' in str(e)
        return
    raise AssertionError('debía negarse a arrancar con e2e.enabled=false')


def test_smoke_train_and_infer():
    """2 iteraciones de entrenamiento + inferencia sobre una caché sintética mínima (CPU)."""
    from tracking.tools.make_synthetic import make
    from tracking.e2e.train import main
    from tracking.e2e import infer
    import argparse
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(0)
        for name in ('tr00', 'tr01'):
            make(f'{tmp}/train', name, 60, 4, 12, 10, rng)
        make(f'{tmp}/val', 'va00', 60, 4, 12, 10, rng)
        a = argparse.Namespace(cfg=None, seed=0, device='cpu', gt_root=None, out=f'{tmp}/run', train_cache=f'{tmp}/train/cache',
                               val_cache=f'{tmp}/val/cache',
                               update=['train.epochs=1', 'train.samples_per_epoch=4', 'train.batch=2', 'train.T=5', 'train.val_samples=2'])
        main(a)
        import json
        for l in open(f'{tmp}/run/log.txt'):
            rec = json.loads(l)
            assert np.isfinite(rec['train_loss']) and np.isfinite(rec['val_loss']), f'pérdida no finita en el registro: {rec}'
        ck = torch.load(f'{tmp}/run/best.pth')
        cfg = load_config(None, ck['cfg'])
        model = E2EModel(cfg, ck['in_dim']).eval(); model.load_state_dict(ck['model'])
        rows, _ = infer.track_sequence(model, torch.load(f'{tmp}/val/cache/va00.pt'), cfg, 'cpu')
        assert isinstance(rows, list)
        assert os.path.isfile(f'{tmp}/run/log.txt')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for t in tests:
        t(); print('ok  ', t.__name__)
    print(f'{len(tests)} tests correctos')
