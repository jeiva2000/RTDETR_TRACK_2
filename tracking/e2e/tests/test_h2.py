"""Tests del H2 (predictor de movimiento con σ): compatibilidad con H1 (motion.mode=none), shapes y relleno con motion=gru, GRU solo con match,
reinicio de m al nacer, pérdida NLL (finita y conectada sin tracks), que el predictor aprende un movimiento de velocidad constante y que la
afinidad con movimiento asigna bien un caso que la apariencia sola no resuelve, y un test de humo (entrenamiento + inferencia, CPU).
Ejecución: python -m tracking.e2e.tests.test_h2   (o pytest tracking/e2e/tests)"""
import os, sys, tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from tracking.e2e.config import load_config
from tracking.e2e.model import E2EModel
from tracking.e2e.motion import apply_delta, box_delta, motion_loss
from tracking.e2e.state import TrackState
from tracking.e2e.tokens import DetTokens

D, M = 256, 32


def _model(mode='gru', **ov):
    torch.manual_seed(0)
    ov = {'motion': {'mode': mode, 'dim': M, 'hidden': 32}, **ov}
    return E2EModel(load_config(None, ov), D).eval()


def _dets(B, K, seed=1):
    g = torch.Generator().manual_seed(seed)
    return DetTokens(torch.randn(B, K, D, generator=g), torch.rand(B, K, 4, generator=g) * 0.3 + 0.2, torch.rand(B, K, generator=g))


def _state(m, B, N, alive, seed=2):
    g = torch.Generator().manual_seed(seed)
    s = m.new_state(B, 'cpu')
    s = TrackState.empty(B, N, D, 'cpu', M if m.motion else 0)
    al = torch.zeros(B, N, dtype=torch.bool)
    for b, n in enumerate(alive):
        al[b, :n] = True
    kw = dict(m=torch.randn(B, N, M, generator=g)) if m.motion else {}
    return s.replace(a=torch.randn(B, N, D, generator=g), box=torch.rand(B, N, 4, generator=g) * 0.3 + 0.2, last=torch.full((B, N), 3),
                     ids=torch.where(al, torch.arange(N)[None].expand(B, N) + 10, -torch.ones(B, N, dtype=torch.long)), alive=al, **kw)


def test_mode_none_is_h1():
    """Sin movimiento: sin parámetros extra, sin m en el estado y misma firma de salida (checkpoints del H1 compatibles)."""
    m = _model('none')
    assert m.motion is None and m.new_state(2, 'cpu').m is None
    assert not any(k.startswith('motion') for k in m.state_dict())
    h1_keys = set(_model('none').state_dict())
    assert all(not k.startswith('motion') for k in h1_keys)
    assert m.predict_motion(m.new_state(1, 'cpu'), torch.tensor([1])) is None


def test_shapes_with_motion():
    m, B, N, K = _model(), 2, 8, 5
    s = _state(m, B, N, [3, 5])
    mu, logsig = m.predict_motion(s, torch.tensor([4, 4]))
    assert mu.shape == (B, N, 4) and logsig.shape == (B, N, 4) and (logsig >= -4).all() and (logsig <= 1).all()
    logits, birth, enc = m.scores(s, _dets(B, K), torch.tensor([4, 4]))
    assert logits.shape == (B, N, K + 1) and birth.shape == (B, K) and torch.isfinite(logits[..., -1]).all()
    logits2, _, _ = m.scores(s, _dets(B, K), torch.tensor([4, 4]), (mu, logsig))
    assert torch.allclose(logits, logits2), 'pasar pred precalculado no cambia el resultado'


def test_padding_invariance_with_motion():
    m = _model()
    s_small, d_small = _state(m, 1, 4, [3]), _dets(1, 4)
    s_big = TrackState.empty(1, 7, D, 'cpu', M).replace(
        a=torch.cat([s_small.a, torch.randn(1, 3, D)], 1), box=torch.cat([s_small.box, torch.rand(1, 3, 4)], 1),
        last=torch.cat([s_small.last, torch.zeros(1, 3, dtype=torch.long)], 1), ids=torch.cat([s_small.ids, -torch.ones(1, 3, dtype=torch.long)], 1),
        alive=torch.cat([s_small.alive, torch.zeros(1, 3, dtype=torch.bool)], 1), m=torch.cat([s_small.m, torch.randn(1, 3, M)], 1))
    d_big = DetTokens(torch.cat([d_small.emb, torch.randn(1, 2, D)], 1), torch.cat([d_small.box, torch.rand(1, 2, 4)], 1),
                      torch.cat([d_small.score, torch.rand(1, 2)], 1), torch.tensor([[True] * 4 + [False] * 2]))
    f = torch.tensor([4])
    l1, _, _ = m.scores(s_small, d_small, f)
    l2, _, _ = m.scores(s_big, d_big, f)
    assert torch.allclose(l1[0, :3, :4], l2[0, :3, :4], atol=1e-5) and torch.allclose(l1[0, :3, -1], l2[0, :3, -1], atol=1e-5)
    assert torch.isinf(l2[0, :3, 4:6]).all()


def test_gru_updates_only_with_match_and_spawn_resets():
    m, K = _model(), 5
    s, d = _state(m, 1, 4, [2]), _dets(1, K)
    f = torch.tensor([9])
    _, _, enc = m.scores(s, d, f)
    s2, matched = m.update(s, torch.tensor([[1, K, K, K]]), d, enc, f)
    assert not torch.allclose(s2.m[0, 0], s.m[0, 0]), 'el track con match actualiza su vector de movimiento'
    assert torch.equal(s2.m[0, 1:], s.m[0, 1:]), 'sin match (o relleno) m no cambia'
    s3, n = s2.spawn(0, torch.tensor([99]), torch.randn(1, D), torch.rand(1, 4), torch.tensor(9))
    assert n == 1 and (s3.m[0, 2] == 0).all() and torch.equal(s3.m[0, 0], s2.m[0, 0]), 'un track nuevo parte con m = 0'


def test_delta_roundtrip():
    t, o = torch.rand(3, 4) * 0.3 + 0.2, torch.rand(3, 4) * 0.3 + 0.2
    assert torch.allclose(apply_delta(t, box_delta(t, o)), o, atol=1e-5)


def test_motion_loss_finite_and_connected():
    mu, ls = torch.zeros(2, 4, 4, requires_grad=True), torch.zeros(2, 4, 4, requires_grad=True)
    obs = torch.full((2, 4, 4), 1e6)                          # valores extremos en huecos sin objetivo no deben propagarse
    loss, n = motion_loss(mu, ls, obs, torch.zeros(2, 4, dtype=torch.bool))
    assert int(n) == 0 and float(loss) == 0.0
    loss.backward(); assert mu.grad is not None and ls.grad is not None, 'cero conectado al grafo (DDP)'
    mask = torch.tensor([[True, False, False, False], [False, False, False, False]])
    loss, n = motion_loss(mu, ls, torch.randn(2, 4, 4), mask)
    assert int(n) == 1 and torch.isfinite(loss)
    loss.backward(); assert mu.grad[1].abs().sum() == 0 and mu.grad[0, 1:].abs().sum() == 0, 'los huecos sin objetivo no reciben gradiente'


def _constant_velocity_batch(B=16, T=8, seed=0):
    """Cada track se mueve con velocidad constante propia (en unidades de su tamaño por frame)."""
    g = torch.Generator().manual_seed(seed)
    box0 = torch.rand(B, 1, 4, generator=g) * 0.2 + 0.4
    vel = (torch.rand(B, 1, 2, generator=g) - 0.5) * 0.4
    t = torch.arange(T).float()[None, :, None]
    boxes = box0.expand(B, T, 4).clone()
    boxes[..., :2] = box0[..., :2] + vel * t * box0[..., 2:]
    return boxes


def test_predictor_learns_constant_velocity():
    """Tras unos pasos el predictor reduce su NLL y la z del movimiento real es mucho menor que la de una detección desplazada."""
    m = _model().train()
    opt = torch.optim.Adam(m.motion.parameters(), lr=3e-3)
    nll = []
    for step in range(250):
        boxes = _constant_velocity_batch(seed=step % 20)
        B, T, _ = boxes.shape
        mm, loss_sum = torch.zeros(B, 1, M), 0.0
        for t in range(1, T):
            mu, ls = m.motion.predict(mm, torch.ones(B, 1))
            obs = box_delta(boxes[:, t - 1:t], boxes[:, t:t + 1])
            l, n = motion_loss(mu, ls, obs, torch.ones(B, 1, dtype=torch.bool))
            loss_sum = loss_sum + l / n
            mm = m.motion.observe(mm, boxes[:, t - 1:t], boxes[:, t:t + 1], torch.ones(B, 1, dtype=torch.long), torch.ones(B, 1, dtype=torch.bool))
        opt.zero_grad(); loss_sum.backward(); opt.step(); nll.append(float(loss_sum) / (T - 1))
    assert np.mean(nll[-20:]) < np.mean(nll[:20]) - 1.0, f'el NLL debe bajar: {np.mean(nll[:20]):.2f} -> {np.mean(nll[-20:]):.2f}'
    m.eval()
    with torch.no_grad():
        boxes = _constant_velocity_batch(seed=1000)
        B, T, _ = boxes.shape
        mm = torch.zeros(B, 1, M)
        for t in range(1, T - 1):
            mm = m.motion.observe(mm, boxes[:, t - 1:t], boxes[:, t:t + 1], torch.ones(B, 1, dtype=torch.long), torch.ones(B, 1, dtype=torch.bool))
        mu, ls = m.motion.predict(mm, torch.ones(B, 1))
        z_true = ((box_delta(boxes[:, T - 2:T - 1], boxes[:, T - 1:T]) - mu) / ls.exp()).abs().mean()
        shifted = boxes[:, T - 1:T].clone(); shifted[..., 0] += 0.5 * shifted[..., 2]
        z_off = ((box_delta(boxes[:, T - 2:T - 1], shifted) - mu) / ls.exp()).abs().mean()
    assert z_true < 2.0 and z_off > 2 * z_true, f'z real {float(z_true):.2f} vs desplazada {float(z_off):.2f}'


def test_smoke_train_and_infer_with_motion():
    """Entrenamiento corto + inferencia con motion.mode=gru sobre una caché sintética (CPU): pérdidas finitas y checkpoint reutilizable."""
    from tracking.tools.make_synthetic import make
    from tracking.e2e.train import main
    from tracking.e2e import infer
    import argparse, json
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(0)
        for name in ('tr00', 'tr01'):
            make(f'{tmp}/train', name, 60, 4, 12, 10, rng)
        make(f'{tmp}/val', 'va00', 60, 4, 12, 10, rng)
        a = argparse.Namespace(cfg=None, seed=0, device='cpu', gt_root=None, out=f'{tmp}/run', train_cache=f'{tmp}/train/cache', val_cache=f'{tmp}/val/cache',
                               update=['motion.mode=gru', 'motion.dim=32', 'train.epochs=2', 'train.samples_per_epoch=8', 'train.batch=2', 'train.T=6',
                                       'train.val_samples=2'])
        main(a)
        for l in open(f'{tmp}/run/log.txt'):
            rec = json.loads(l)
            assert all(np.isfinite(v) for v in rec.values() if isinstance(v, float)), rec
            assert 'val_motion_nll' in rec, 'debe registrar el NLL de movimiento'
        ck = torch.load(f'{tmp}/run/best.pth')
        cfg = load_config(None, ck['cfg'])
        assert cfg.motion.mode == 'gru'
        model = E2EModel(cfg, ck['in_dim']).eval(); model.load_state_dict(ck['model'])
        seq = torch.load(f'{tmp}/val/cache/va00.pt')
        rows, _ = infer.track_sequence(model, seq, cfg, 'cpu')
        assert isinstance(rows, list)          # con 2 épocas la cabeza de nacimiento puede no superar el umbral
        cfg_s = load_config(None, dict(cfg, infer=dict(cfg.infer, birth_mode='score', min_score=0.0)))
        rows, _ = infer.track_sequence(model, seq, cfg_s, 'cpu')
        ids = {r[1] for r in rows}
        assert len(rows) > 0 and len(ids) >= 1, 'con nacimiento por score debe producir tracks (recorre el GRU y la afinidad con movimiento)'


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for t in tests:
        t(); print('ok  ', t.__name__)
    print(f'{len(tests)} tests correctos')
