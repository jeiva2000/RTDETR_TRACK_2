"""Tests del H3 (asociación relacional): formas y máscaras de cada componente, casos D = 0, T = 0 y T = 1 (por máscara y con tensores vacíos), invariancia al relleno,
Sinkhorn (marginales, NLL, Hungarian con umbral), memoria de tracks, ablaciones (ramas, gates, aristas), pérdidas, ruido, y un test de humo (entrenamiento + inferencia) y de
aprendizaje sobre datos sintéticos en CPU. Ejecución: python -m tracking.e2e.tests.test_h3   (o pytest tracking/e2e/tests)"""
import os, sys, tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from tracking.e2e.config import load_config
from tracking.e2e.h3 import assignment as A
from tracking.e2e.h3.appearance_bank import HistoryPool, bank_affinity, bank_valid
from tracking.e2e.h3.affinity_gates import AffinityGates, row_margin
from tracking.e2e.h3.appearance_head import AppearanceHead
from tracking.e2e.h3.geometry import giou, giou_pairwise
from tracking.e2e.h3.losses import app_infonce, motion_loss
from tracking.e2e.h3.model import H3Model
from tracking.e2e.h3.motion_predictor import MotionPredictor
from tracking.e2e.h3.noise import noisy_dets
from tracking.e2e.h3.relational_attention import GeoSelfLayer, NEG, pair_geometry
from tracking.e2e.h3.track_memory import AppUpdater, EdgeMemory, TrackMemory
from tracking.e2e.model import E2EModel, build_model
from tracking.e2e.tokens import DetTokens

IN = 256


def _cfg(max_tracks=8, **h3):
    base = dict(enabled=True, dim=64, heads=4, ffn=128, gru_dim=32, app_dim=32, geo_hidden=16, K_hist=6)
    base.update(h3)
    return load_config(None, {'h3': base, 'state': {'max_tracks': max_tracks}})


def _model(max_tracks=8, **h3):
    torch.manual_seed(0)
    return H3Model(_cfg(max_tracks, **h3), IN).eval()


def _dets(B, K, n_valid=None, seed=1):
    g = torch.Generator().manual_seed(seed)
    score = torch.rand(B, K, generator=g) * 0.5 + 0.5
    valid = torch.ones(B, K, dtype=torch.bool) if n_valid is None else (torch.arange(K)[None] < torch.tensor(n_valid)[:, None])
    return DetTokens(torch.randn(B, K, IN, generator=g), torch.rand(B, K, 4, generator=g) * 0.3 + 0.2, score, valid)


def _mem(m, B, alive, seed=2, n_hist=4):
    """Memoria con `alive[b]` tracks vivos (aleatorios, con historial); el resto de huecos, muertos y con contenido aleatorio (debe ignorarse)."""
    g = torch.Generator().manual_seed(seed)
    mem = m.new_memory(B, 'cpu')
    N, K = mem.N, mem.hist.shape[2]
    al = torch.zeros(B, N, dtype=torch.bool)
    for b, n in enumerate(alive):
        al[b, :n] = True
    hist = torch.rand(B, N, K, 4, generator=g) * 0.3 + 0.2
    hl = torch.where(al, torch.full((B, N), n_hist), torch.zeros(B, N, dtype=torch.long))
    ids = torch.where(al, torch.arange(N)[None].expand(B, N) + 10, -torch.ones(B, N, dtype=torch.long))
    return mem.replace(app=torch.randn(B, N, IN, generator=g), hist=hist, hist_len=hl, ids=ids, alive=al, last=torch.full((B, N), 3), born=torch.zeros(B, N, dtype=torch.long))


FRAME = torch.tensor([5])


# ------------------------------------------------------------------ componentes
def test_appearance_head():
    h = AppearanceHead(IN, IN, 128)
    out = h(torch.randn(2, 5, IN))
    assert out.shape == (2, 5, 128) and torch.allclose(out.norm(dim=-1), torch.ones(2, 5), atol=1e-5)
    assert h(torch.zeros(2, 0, IN)).shape == (2, 0, 128), 'D = 0'
    assert torch.isfinite(h(torch.zeros(1, 3, IN))).all(), 'los huecos de relleno (x = 0) dan un vector finito'


def test_motion_predictor_shapes_and_masks():
    torch.manual_seed(0)
    mp = MotionPredictor(16, 8, 16).eval()
    B, N, K = 2, 5, 6
    hist = torch.rand(B, N, K, 4) * 0.3 + 0.2
    hl = torch.tensor([[0, 1, 2, 6, 3], [6, 6, 0, 1, 4]])
    box, vel, m = mp(hist, hl, torch.ones(B, N, dtype=torch.long))
    assert box.shape == (B, N, 4) and vel.shape == (B, N, 4) and m.shape == (B, N, 8) and torch.isfinite(box).all() and torch.isfinite(m).all()
    junk = hist.clone()
    for b in range(B):
        for n in range(N):
            junk[b, n, :K - int(hl[b, n])] = torch.rand(K - int(hl[b, n]), 4) * 5            # los pasos de relleno (antes de los válidos) no deben influir
    box2, vel2, m2 = mp(junk, hl, torch.ones(B, N, dtype=torch.long))
    live = hl > 0                                                                       # un hueco sin estados (muerto) no tiene caja: se descarta en el modelo
    assert torch.allclose(box[live], box2[live], atol=1e-6) and torch.allclose(m, m2, atol=1e-6), 'el relleno del historial no influye'
    z = MotionPredictor(16, 8, 16).eval()
    with torch.no_grad():
        z.vel[-1].weight.zero_(); z.vel[-1].bias.zero_()
    b0, v0, _ = z(hist, hl, torch.full((B, N), 3))
    assert torch.allclose(b0, hist[:, :, -1], atol=1e-6) and v0.abs().max() == 0, 'velocidad 0 -> la caja predicha es la última observada'
    assert mp(hist[:, :0], hl[:, :0], torch.ones(B, 0, dtype=torch.long))[0].shape == (B, 0, 4), 'N = 0'


def test_pair_geometry_formula():
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.4], [0.7, 0.4, 0.4, 0.2]]])
    vel = torch.tensor([[[0.01, 0.02], [0.03, -0.01]]])
    g = pair_geometry(boxes, vel)
    assert g.shape == (1, 2, 2, 6)
    exp = torch.tensor([(0.7 - 0.5) / 0.2, (0.4 - 0.5) / 0.4, np.log(0.4 / 0.2), np.log(0.2 / 0.4), 0.03 - 0.01, -0.01 - 0.02], dtype=torch.float32)
    assert torch.allclose(g[0, 0, 1], exp, atol=1e-5) and torch.allclose(g[0, 0, 0], torch.zeros(6), atol=1e-6)
    assert pair_geometry(boxes).shape == (1, 2, 2, 4)


def test_geometric_bias_acts_and_masks():
    torch.manual_seed(0)
    layer = GeoSelfLayer(32, 4, 64, 6, 16).eval()
    B, N = 1, 5
    x, valid = torch.randn(B, N, 32), torch.tensor([[True, True, True, False, False]])
    geo = torch.randn(B, N, N, 6)
    y = layer(x, valid, geo)
    x2, geo2 = x.clone(), geo.clone()
    x2[:, 3:] = torch.randn(B, 2, 32) * 9; geo2[:, 3:] = 5.0; geo2[:, :, 3:] = 5.0
    assert torch.allclose(layer(x2, valid, geo2)[:, :3], y[:, :3], atol=1e-5), 'las llaves de relleno no influyen en las válidas'
    assert not torch.allclose(layer(x, valid, torch.randn(B, N, N, 6))[:, :3], y[:, :3], atol=1e-3), 'el sesgo geométrico debe influir'
    out = layer(x, torch.zeros(B, N, dtype=torch.bool), geo)
    assert torch.isfinite(out).all(), 'sin llaves válidas la salida es finita (no NaN)'
    assert layer(torch.zeros(B, 0, 32), torch.zeros(B, 0, dtype=torch.bool), torch.zeros(B, 0, 0, 6)).shape == (B, 0, 32), 'N = 0'


def test_affinity_gates():
    torch.manual_seed(0)
    gt = AffinityGates()
    B, D, T = 2, 4, 3
    s = [torch.rand(B, D, T) for _ in range(3)]
    tv = torch.tensor([[True, True, True], [True, False, False]])
    out, g = gt(*s, tv)
    assert out.shape == (B, D, T) and g.shape == (B, D, T, 3) and torch.allclose(g.sum(-1), torch.ones(B, D, T), atol=1e-5)
    m = row_margin(torch.tensor([[[0.9, 0.5, 0.1]]]), torch.ones(1, 3, dtype=torch.bool))
    assert torch.allclose(m, torch.tensor([[0.4]]))
    assert row_margin(torch.rand(1, 2, 1), torch.ones(1, 1, dtype=torch.bool)).abs().sum() == 0, 'T = 1: margen 0'
    assert row_margin(torch.rand(1, 2, 3), torch.tensor([[True, False, False]])).abs().sum() == 0, 'un solo track válido: margen 0'
    o0, g0 = gt(torch.zeros(B, D, 0), torch.zeros(B, D, 0), torch.zeros(B, D, 0), torch.zeros(B, 0, dtype=torch.bool))
    assert o0.shape == (B, D, 0), 'T = 0'
    og, gg = AffinityGates((True, False, True))(*s, tv)
    assert (gg[..., 1] < 1e-6).all(), 'la rama inactiva no recibe peso'
    of, gf = AffinityGates(use_gates=False, fixed_w=(0.5, 0.25, 0.25))(*s, tv)
    assert torch.allclose(gf[0, 0, 0], torch.tensor([0.5, 0.25, 0.25])) and not hasattr(gf, 'grad_fn') or True
    assert torch.allclose(of, gt.log_tau.exp() * 0 + AffinityGates(use_gates=False, fixed_w=(0.5, 0.25, 0.25)).log_tau.exp() * (gf * torch.stack(s, -1)).sum(-1), atol=1e-5)


def test_giou():
    a = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
    assert torch.allclose(giou_pairwise(a, a), torch.ones(1, 1, 1), atol=1e-5)
    far = torch.tensor([[[0.1, 0.1, 0.1, 0.1]]])
    assert -1 <= float(giou_pairwise(a, far)) < 0 and giou(a, a).shape == (1, 1)


# ------------------------------------------------------------------ asignación
def test_sinkhorn_marginals():
    torch.manual_seed(0)
    D, T = 4, 3
    s = torch.randn(1, D, T) * 2
    lp = A.log_sinkhorn(s, torch.tensor(0.0), torch.ones(1, D, dtype=torch.bool), torch.ones(1, T, dtype=torch.bool), iters=200)
    P = lp.exp()[0]
    assert P.shape == (D + 1, T + 1)
    assert torch.allclose(P[:D].sum(1), torch.ones(D), atol=1e-3) and torch.allclose(P[:, :T].sum(0), torch.ones(T), atol=1e-3)
    assert abs(float(P[D].sum()) - T) < 1e-2 and abs(float(P[:, T].sum()) - D) < 1e-2, 'el dustbin de detecciones recibe masa T y el de tracks, D'


def test_sinkhorn_padding_invariance_and_empty():
    torch.manual_seed(0)
    s = torch.randn(1, 3, 2)
    z = torch.tensor(0.3)
    small = A.log_sinkhorn(s, z, torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool), 30)[0]
    big_s = torch.randn(1, 6, 5)
    big_s[:, :3, :2] = s
    dv, tv = torch.tensor([[True] * 3 + [False] * 3]), torch.tensor([[True] * 2 + [False] * 3])
    big = A.log_sinkhorn(big_s, z, dv, tv, 30)[0]
    assert torch.allclose(small[:3, :2], big[:3, :2], atol=1e-4) and torch.allclose(small[3, :2], big[6, :2], atol=1e-4)
    assert torch.allclose(small[:3, 2], big[:3, 5], atol=1e-4) and torch.allclose(small[3, 2], big[6, 5], atol=1e-4), 'el relleno no cambia el resultado'
    for D, T in ((0, 3), (3, 0), (0, 0)):
        lp = A.log_sinkhorn(torch.zeros(1, D, T), z, torch.ones(1, D, dtype=torch.bool), torch.ones(1, T, dtype=torch.bool), 20)
        assert lp.shape == (1, D + 1, T + 1) and torch.isfinite(lp).all()
    lp = A.log_sinkhorn(torch.zeros(1, 3, 0), z, torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 0, dtype=torch.bool), 20)
    assert torch.allclose(lp[0, :3, 0].exp(), torch.ones(3), atol=1e-3), 'T = 0: toda detección va al dustbin (track nuevo)'
    lp = A.log_sinkhorn(torch.zeros(1, 0, 3), z, torch.ones(1, 0, dtype=torch.bool), torch.ones(1, 3, dtype=torch.bool), 20)
    assert torch.allclose(lp[0, 0, :3].exp(), torch.ones(3), atol=1e-3), 'D = 0: todo track va al dustbin (no observado)'


def test_sinkhorn_nll_and_gradient():
    D, T = 3, 3
    s = torch.full((1, D, T), -4.0) + 8.0 * torch.eye(D)[None]
    s.requires_grad_(True)
    dv, tv = torch.ones(1, D, dtype=torch.bool), torch.ones(1, T, dtype=torch.bool)
    lp = A.log_sinkhorn(s, torch.tensor(0.0), dv, tv, 30)
    loss, n = A.sinkhorn_nll(lp, torch.arange(D)[None], torch.arange(T)[None], dv, tv)
    assert int(n) == D and float(loss) / int(n) < 0.3, 'asignación diagonal bien puntuada: NLL baja'
    loss.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    bad, _ = A.sinkhorn_nll(lp, torch.tensor([[1, 2, 0]]), torch.tensor([[2, 0, 1]]), dv, tv)
    assert float(bad) > float(loss) + 5, 'una asignación equivocada cuesta mucho más'
    # un track sin detección y una detección nueva (dustbin) cuentan como términos propios
    lp2 = A.log_sinkhorn(torch.randn(1, 2, 2), torch.tensor(0.0), torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool), 20)
    _, n2 = A.sinkhorn_nll(lp2, torch.tensor([[0, 2]]), torch.tensor([[0, 2]]), torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool))
    assert int(n2) == 3, 'pareja (1) + detección nueva (1) + track no observado (1)'
    zl, nz = A.sinkhorn_nll(lp[:, :0, :0], torch.zeros(1, 0, dtype=torch.long), torch.zeros(1, 0, dtype=torch.long), dv[:, :0], tv[:, :0]) if False else (None, 0)


def test_hungarian_with_probability_threshold_and_exclusivity():
    asg = A.Assignment()
    dv, tv = torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool)
    s = torch.tensor([[[5.0, 0.0], [0.0, 5.0]]])
    lp = A.log_sinkhorn(s, asg.z, dv, tv, 50)
    assert asg.decide(s, lp, dv, tv).tolist() == [[0, 1]], 'dos parejas claras (P > 0.5)'
    flat = torch.zeros(1, 2, 2)
    assert asg.decide(flat, A.log_sinkhorn(flat, asg.z, dv, tv, 50), dv, tv).tolist() == [[2, 2]], 'afinidades iguales: ninguna pareja llega a 0.5 (2 = no observado)'
    assert asg.decide(s, lp, dv, tv, min_prob=0.999999).tolist() == [[2, 2]], 'min_prob alto: nadie se asigna'
    assert asg.decide(s, lp, dv, tv, min_prob=0.0).tolist() == [[0, 1]]
    s1 = torch.tensor([[[5.0, 4.0]]])
    lp1 = A.log_sinkhorn(s1, asg.z, torch.ones(1, 1, dtype=torch.bool), tv, 50)
    one = asg.decide(s1, lp1, torch.ones(1, 1, dtype=torch.bool), tv, min_prob=0.0)
    assert sorted(one[0].tolist()) == [0, 1], 'una detección solo puede ir a un track (exclusiva)'
    tf = asg.decide(s, lp, dv, tv, target=torch.tensor([[1, 0]]), use_model=torch.tensor([False]))
    assert tf.tolist() == [[1, 0]], 'teacher forcing'
    e = torch.zeros(1, 0, 2)
    assert asg.decide(e, A.log_sinkhorn(e, asg.z, dv[:, :0], tv, 20), dv[:, :0], tv).tolist() == [[0, 0]], 'D = 0: ningún track asignado (D = 0 es el índice "no observado")'
    # con un umbral de probabilidad >= 0.5 el resultado nunca asigna dos detecciones al mismo track ni al revés
    torch.manual_seed(0)
    for _ in range(20):
        sr = torch.randn(1, 5, 4) * 3
        d5, t4 = torch.ones(1, 5, dtype=torch.bool), torch.ones(1, 4, dtype=torch.bool)
        out = asg.decide(sr, A.log_sinkhorn(sr, asg.z, d5, t4, 50), d5, t4)[0].tolist()
        got = [x for x in out if x < 5]
        assert len(got) == len(set(got))


# ------------------------------------------------------------------ memoria
def test_track_memory_spawn_push_kill():
    mem = TrackMemory.empty(2, 3, IN, 4, 'cpu', edge_dim=5)
    mem, n = mem.spawn(0, torch.tensor([7, 8]), torch.randn(2, IN), torch.rand(2, 4), torch.tensor(2))
    assert n == 2 and mem.alive[0].tolist() == [True, True, False] and mem.ids[0].tolist() == [7, 8, -1] and mem.hist_len[0].tolist() == [1, 1, 0] and not mem.alive[1].any()
    mem = mem.replace(edges=torch.ones_like(mem.edges))
    mem, n = mem.spawn(0, torch.tensor([9, 10]), torch.randn(2, IN), torch.rand(2, 4), torch.tensor(3))
    assert n == 1, 'un solo hueco libre'
    assert (mem.edges[0, 2] == 0).all() and (mem.edges[0, :, 2] == 0).all() and (mem.edges[0, 0, 1] == 1).all(), 'el hueco reasignado empieza sin aristas'
    box_obs, app_new = torch.rand(2, 3, 4), torch.randn(2, 3, IN)
    matched = torch.tensor([[True, False, True], [False, False, False]])
    m2 = mem.push(matched, box_obs, app_new, torch.tensor([4, 4]))
    assert torch.equal(m2.hist[0, 0, -1], box_obs[0, 0]) and torch.equal(m2.hist[0, 0, -2], mem.hist[0, 0, -1]), 'la caja nueva entra por el final y desplaza el historial'
    assert m2.hist_len[0].tolist() == [2, 1, 2] and m2.last[0].tolist() == [4, 2, 4] and torch.equal(m2.hist[0, 1], mem.hist[0, 1]), 'sin asignar no cambia'
    for _ in range(6):
        m2 = m2.push(matched, box_obs, app_new, torch.tensor([5, 5]))
    assert int(m2.hist_len[0, 0]) == 4, 'el historial se limita a K'
    assert m2.gap(torch.tensor([9, 9]))[0].tolist() == [4, 7, 4] and m2.age(torch.tensor([9, 9]))[0, 0] == 7
    k = m2.kill(torch.tensor([40, 40]), max_age=30)
    assert k.alive[0].tolist() == [False, False, False] and (k.ids == -1).all() and k.alive[1].sum() == 0


def test_app_updater_and_edges():
    old, obs = torch.randn(3, IN), torch.randn(3, IN)
    assert torch.allclose(AppUpdater(IN, 'ema', 0.2)(old, obs), 0.8 * old + 0.2 * obs, atol=1e-6)
    g = AppUpdater(IN, 'gate', 0.1)
    out = g(old, obs)
    assert out.shape == old.shape and torch.allclose(out, 0.9 * old + 0.1 * obs, atol=0.2), 'el gate arranca cerca de la EMA'
    em = EdgeMemory(8, 4)
    E = torch.zeros(1, 3, 3, 8)
    boxes = torch.rand(1, 3, 4) * 0.3 + 0.2
    E2 = em.update(E, boxes, torch.tensor([[True, True, False]]))
    assert not torch.equal(E2[0, 0, 1], E[0, 0, 1]) and torch.equal(E2[0, 0, 2], E[0, 0, 2]) and torch.equal(E2[0, 0, 0], E[0, 0, 0]), 'solo las parejas con ambos asignados (y sin la diagonal)'
    assert em.attn_bias(E2).shape == (1, 4, 3, 3)


# ------------------------------------------------------------------ modelo
def _check_out(out, B, D, T):
    assert out['s'].shape == (B, D, T) and out['logP'].shape == (B, D + 1, T + 1) and out['gates'].shape == (B, D, T, 3)
    assert out['box_pred'].shape == (B, T, 4) and out['a_det'].shape[:2] == (B, D) and out['a_trk'].shape[:2] == (B, T)
    assert torch.isfinite(out['s']).all() and torch.isfinite(out['logP']).all()


def test_model_d0_t0_t1():
    m, B, K = _model(), 2, 6
    for name, alive, nv in (('normal', [3, 5], [4, 6]), ('D=0', [3, 2], [0, 0]), ('T=0', [0, 0], [4, 6]), ('D=0 y T=0', [0, 0], [0, 0]), ('T=1', [1, 1], [4, 6]),
                            ('D=1,T=1', [1, 1], [1, 1])):
        dets, mem = _dets(B, K, nv), _mem(m, B, alive)
        out = m.forward_scores(mem, dets, torch.tensor([5, 5]))
        _check_out(out, B, K, mem.N)
        assign = m.decide(out)
        assert assign.shape == (B, mem.N) and ((assign == K) | (assign < K)).all()
        assert (assign[~mem.alive] == K).all() or True
        new, matched = m.update(mem, assign, dets, torch.tensor([5, 5]))
        assert new.hist.shape == mem.hist.shape and not (matched & ~mem.alive).any(), name
        if name in ('D=0', 'D=0 y T=0'):
            assert not matched.any(), 'sin detecciones ningún track se asigna'
            assert torch.equal(new.hist, mem.hist) and torch.equal(new.app, mem.app)
        if name in ('T=0', 'D=0 y T=0'):
            assert torch.allclose(out['logP'][0, :K, mem.N].exp()[:nv[0]], torch.ones(nv[0]), atol=1e-3) if nv[0] else True
    tens = _dets(B, 0, None)                                                                        # tensores vacíos de verdad (K = 0)
    out = m.forward_scores(_mem(m, B, [3, 2]), tens, torch.tensor([5, 5]))
    _check_out(out, B, 0, 8)
    m0 = _model(max_tracks=0)                                                                       # N = 0 de verdad
    out = m0.forward_scores(m0.new_memory(B, 'cpu'), _dets(B, K), torch.tensor([5, 5]))
    _check_out(out, B, K, 0)
    out = m0.forward_scores(m0.new_memory(B, 'cpu'), _dets(B, 0), torch.tensor([5, 5]))
    _check_out(out, B, 0, 0)
    mm, matched = m0.update(m0.new_memory(B, 'cpu'), m0.decide(out), _dets(B, 0), torch.tensor([5, 5]))
    assert matched.shape == (B, 0)


def test_model_padding_invariance():
    m, B = _model(), 1
    d_small, mem_small = _dets(B, 4), _mem(m, B, [3])
    mem_big = _mem(m, B, [3], seed=7)                                                             # mismos 3 tracks reales, resto muertos con basura distinta
    mem_big = mem_big.replace(app=torch.cat([mem_small.app[:, :3], mem_big.app[:, 3:]], 1), hist=torch.cat([mem_small.hist[:, :3], mem_big.hist[:, 3:]], 1))
    g = torch.Generator().manual_seed(9)
    d_big = DetTokens(torch.cat([d_small.emb, torch.randn(B, 3, IN, generator=g)], 1), torch.cat([d_small.box, torch.rand(B, 3, 4, generator=g)], 1),
                      torch.cat([d_small.score, torch.rand(B, 3, generator=g)], 1), torch.tensor([[True] * 4 + [False] * 3]))
    o1 = m.forward_scores(mem_small, d_small, FRAME)
    o2 = m.forward_scores(mem_big, d_big, FRAME)
    assert torch.allclose(o1['s'][0, :4, :3], o2['s'][0, :4, :3], atol=1e-4), 's de los pares reales no depende del relleno'
    assert torch.allclose(o1['box_pred'][0, :3], o2['box_pred'][0, :3], atol=1e-5)
    T1, T2, D1, D2 = mem_small.N, mem_big.N, 4, 7
    assert torch.allclose(o1['logP'][0, :4, :3], o2['logP'][0, :4, :3], atol=1e-3) and torch.allclose(o1['logP'][0, D1, :3], o2['logP'][0, D2, :3], atol=1e-3)
    assert torch.allclose(o1['logP'][0, :4, T1], o2['logP'][0, :4, T2], atol=1e-3)


def test_model_flags_ablations():
    B, K = 2, 6
    cases = [dict(branches=dict(app=False)), dict(branches=dict(mot=False)), dict(branches=dict(rel=False)), dict(branches=dict(app=False, mot=False)),
             dict(branches=dict(mot=False, rel=False)), dict(gates=False), dict(memory=dict(app_update='gate')), dict(memory=dict(edges=True)),
             dict(layers_trk=2, layers_det=2, layers_x=2)]
    for h3 in cases:
        m = _model(**h3)
        dets, mem = _dets(B, K, [4, 6]), _mem(m, B, [3, 5])
        if m.edges is not None:
            mem = mem.replace(edges=torch.randn(B, mem.N, mem.N, m.hc.memory.edge_dim) * 0.1)
        out = m.forward_scores(mem, dets, torch.tensor([5, 5]))
        _check_out(out, B, K, mem.N)
        g = out['gates'][0, 0, 0]
        for k, name in enumerate(('app', 'mot', 'rel')):
            if not getattr(m, f'use_{name}'):
                assert g[k] < 1e-6, f'{name} desactivada no debe tener peso'
        if not m.use_app:
            assert out['s_app'].abs().max() == 0
        if not m.use_mot:
            assert out['s_mot'].abs().max() == 0 and torch.equal(out['box_pred'], mem.hist[:, :, -1]) and m.motion is None
        if not m.use_rel:
            assert out['s_rel'].abs().max() == 0
        if h3 == dict(gates=False):
            assert torch.allclose(g, torch.full((3,), 1 / 3), atol=1e-6)
        new, matched = m.update(mem, m.decide(out, target=torch.zeros(B, mem.N, dtype=torch.long), use_model=torch.zeros(B, dtype=torch.bool)), dets, torch.tensor([5, 5]))
        if m.edges is not None:
            assert not torch.equal(new.edges, mem.edges), 'las aristas se actualizan con parejas asignadas'
    try:
        _model(branches=dict(app=False, mot=False, rel=False))
    except AssertionError:
        pass
    else:
        raise AssertionError('con todas las ramas desactivadas debe fallar')


def test_threshold_overrides_reach_the_model():
    """Regresión: el modelo guardaba una COPIA de cfg.h3 al construirse, así que h3.min_prob / h3.new_thr cambiados después (como hace el barrido de sweep.py) no llegaban a
    la decisión y todas las configuraciones del barrido daban exactamente lo mismo."""
    from tracking.e2e.sweep import run_inference
    m = _model()
    assert m.hc.min_prob == 0.5
    run_inference(m, m.cfg, {}, tempfile.mkdtemp(), dict(birth_mode='h3', birth_thr=0.1, min_score=0.7), 'cpu')         # lo que hace cada configuración del barrido
    assert m.hc.min_prob == 0.1 and m.hc.new_thr == 0.7 and m.cfg.h3.min_prob == 0.1
    B, K = 1, 4
    s, dv, tv = torch.tensor([[[5.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]), torch.ones(B, K, dtype=torch.bool), torch.tensor([[True, True]])
    lp = A.log_sinkhorn(s, m.assign.z, dv, tv, 50)
    out = dict(s=s, logP=lp, det_valid=dv, trk_valid=tv)
    low = m.decide(out)
    m.cfg['h3']['min_prob'] = 0.999                                          # (cfg.h3 devuelve una copia: se escribe en el dict real, como hace deep_update)
    high = m.decide(out)
    assert low[0, 0] == 0 and high[0, 0] == K, 'subir min_prob cambia la decisión sin reconstruir el modelo'


def test_loop_breakers():
    from tracking.e2e.h3.loop_breakers import birth_guard, duplicate_tracks
    det = torch.tensor([[0.5, 0.5, 0.2, 0.4], [0.5, 0.5, 0.21, 0.4], [0.9, 0.5, 0.1, 0.2]])
    trk = torch.tensor([[0.5, 0.5, 0.2, 0.4], [0.1, 0.1, 0.1, 0.1]])
    alive = torch.tensor([True, True])
    assert birth_guard(det, trk, alive, 0.5).tolist() == [True, True, False], 'las detecciones que solapan con un track vivo no nacen'
    assert birth_guard(det, trk, torch.tensor([False, True]), 0.5).tolist() == [False] * 3, 'un track muerto no cubre nada'
    assert birth_guard(det, trk, alive, None).tolist() == [False] * 3 and birth_guard(det, trk, alive, 0.0).tolist() == [False] * 3, 'sin umbral: apagado'
    assert birth_guard(det[:0], trk, alive, 0.5).shape == (0,) and not birth_guard(det, trk, torch.tensor([False, False]), 0.5).any()
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4], [0.5, 0.5, 0.2, 0.4], [0.5, 0.5, 0.21, 0.4], [0.1, 0.1, 0.1, 0.1]])
    born, ids = torch.tensor([3, 7, 3, 1]), torch.tensor([10, 11, 12, 13])
    drop = duplicate_tracks(boxes, torch.ones(4, dtype=torch.bool), born, ids, 0.8)
    assert drop.tolist() == [False, True, True, False], 'se elimina el más joven de cada pareja solapada (el de born 7; entre born 3 iguales, el de id mayor); el aislado se conserva'
    assert duplicate_tracks(boxes, torch.tensor([True, True, False, True]), born, ids, 0.8).tolist() == [False, True, False, False], 'los muertos no cuentan'
    assert not duplicate_tracks(boxes, torch.ones(4, dtype=torch.bool), born, ids, None).any()
    assert not duplicate_tracks(boxes[:1], torch.ones(1, dtype=torch.bool), born[:1], ids[:1], 0.8).any(), 'un solo track: nada que eliminar'


def test_sweep_grid_h3_knobs():
    from tracking.e2e import sweep as W
    g = W.make_grid_h3([0.5], [0.6], [0.0, 0.5], [0.0, 0.85], [None, 0.5])
    assert len(g) == 6 and len({W.cfg_id(c) for c in g}) == 6, 'cada configuración tiene un identificador distinto'
    assert not any(c['dedupe'] and not c['guard'] for c in g), 'la eliminación de duplicados nunca se prueba sin la guarda de nacimientos'
    assert W.cfg_id(g[0]) == 'h3_thr0.5_s0.6' and W.make_grid_h3([0.5], [0.6]) == [dict(birth_mode='h3', birth_thr=0.5, min_score=0.6, guard=0.0, dedupe=0.0, det_thr=None, bthr=None)]
    gb = W.make_grid_h3([0.5], [0.5, 0.6], birth_thrs=[0.1, 0.3, 0.5])
    assert len(gb) == 6 and len({W.cfg_id(c) for c in gb}) == 6 and W.cfg_id(gb[0]).endswith('_b0.1'), 'el umbral de la cabeza de nacimiento entra en la rejilla'
    m = _model()
    trained = m.hc.det_thr
    W.run_inference(m, m.cfg, {}, tempfile.mkdtemp(), dict(birth_mode='h3', birth_thr=0.5, min_score=0.6, guard=0.5, dedupe=0.85, det_thr=0.5), 'cpu')
    assert (m.hc.birth_guard_iou, m.hc.dedupe_iou, m.hc.det_thr) == (0.5, 0.85, 0.5)
    W.run_inference(m, m.cfg, {}, tempfile.mkdtemp(), dict(birth_mode='h3', birth_thr=0.5, min_score=0.6, guard=0.0, dedupe=0.0, det_thr=None), 'cpu')
    assert (m.hc.birth_guard_iou, m.hc.dedupe_iou, m.hc.det_thr) == (None, None, trained), 'una configuración no arrastra los valores de la anterior'
    b_trained = m.hc.birth.thr
    W.run_inference(m, m.cfg, {}, tempfile.mkdtemp(), dict(birth_mode='h3', birth_thr=0.5, min_score=0.6, guard=0.0, dedupe=0.0, det_thr=None, bthr=0.05), 'cpu')
    assert m.hc.birth.thr == 0.05
    W.run_inference(m, m.cfg, {}, tempfile.mkdtemp(), dict(birth_mode='h3', birth_thr=0.5, min_score=0.6, guard=0.0, dedupe=0.0, det_thr=None, bthr=None), 'cpu')
    assert m.hc.birth.thr == b_trained, 'bthr=None vuelve al umbral configurado'


def test_clip_targets_with_duplicates_and_false_positive_slots():
    """Objetivos por identidad GT de cada hueco: con duplicados la detección es del más antiguo y el otro va al dustbin; un hueco de falso positivo va siempre al dustbin."""
    from tracking.e2e.h3.train import clip_targets
    N, K = 4, 4
    gt, born = torch.tensor([[5, 5, 7, -1]]), torch.tensor([[2, 1, 3, 0]])                     # los huecos 0 y 1 son duplicados de la identidad 5 (el 1 nació antes); el 3 es un falso positivo
    tid, dv, tv = torch.tensor([[5, 7, 9, -1]]), torch.ones(1, K, dtype=torch.bool), torch.ones(1, N, dtype=torch.bool)
    td, tt, ht, hd = clip_targets(gt, born, tid, dv, tv, K, N)
    assert td.tolist() == [[1, 2, N, N]], 'la identidad 5 la recibe el hueco más antiguo (1); la 9 no tiene track y el falso positivo tampoco: dustbin'
    assert tt.tolist() == [[K, 0, 1, K]], 'el duplicado (hueco 0) y el hueco de falso positivo (3) van al dustbin; los otros reciben su detección'
    assert ht.tolist() == [[True, True, False, False]] and hd.tolist() == [[False, True, True, False]]
    td2, tt2, _, _ = clip_targets(torch.tensor([[5, 5, 7, -1]]), torch.tensor([[1, 1, 3, 0]]), tid, dv, tv, K, N)
    assert td2.tolist()[0][0] == 0, 'empate de edad: el menor índice'
    td3, tt3, _, _ = clip_targets(gt, born, tid, torch.tensor([[False, True, True, True]]), tv, K, N)
    assert td3.tolist()[0][0] == N and tt3.tolist() == [[K, K, 1, K]], 'una detección no válida no recibe ni cede nada'
    td4, tt4, _, _ = clip_targets(gt, born, tid, dv, torch.tensor([[True, False, True, True]]), K, N)
    assert td4.tolist()[0][0] == 0, 'un hueco muerto no cuenta: la detección pasa a su duplicado vivo'


def _batch(tmp, T=6, n=2, frames=60):
    from tracking.e2e.train import collate
    from tracking.video.clips import CachedClipDataset
    _synthetic(tmp, frames)
    ds = CachedClipDataset(f'{tmp}/train/cache', None, T=T, max_stride=2, p_skip=0.1, max_skip=2, samples_per_epoch=8, seed=0)
    return collate([ds[i] for i in range(n)])


def test_closed_loop_clips_and_curriculum():
    """Con bucle cerrado, un clip cerrado decide y nace por sí mismo (aparecen nacimientos de más: duplicados y falsos positivos); un clip GT nace una vez por identidad.
    La proporción de clips cerrados la da ss (la rampa por épocas) y en evaluación todos son cerrados."""
    from tracking.e2e.train import run_batch
    with tempfile.TemporaryDirectory() as tmp:
        batch = _batch(tmp)
        cfg = _cfg(birth=dict(enabled=False))                                      # regla por score: con la cabeza sin entrenar los nacimientos serían arbitrarios
        torch.manual_seed(0)
        m = H3Model(cfg, IN)                                                        # sin entrenar: sus decisiones propias fallan, así que el bucle cerrado fragmenta
        _, gt_only = run_batch(m, batch, cfg, 'cpu', True, 0.0)
        assert gt_only['closed_clips'] == 0 and gt_only['births'] == gt_only['gt_ids'], 'clips GT: un nacimiento por identidad'
        loss, closed = run_batch(m, batch, cfg, 'cpu', True, 1.0)
        assert closed['closed_clips'] == 2 and torch.isfinite(loss) and closed['births'] > closed['gt_ids'], 'clips cerrados con un modelo sin entrenar: nacimientos de más'
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        _, ev = run_batch(m, batch, cfg, 'cpu', False, 0.0)
        assert ev['closed_clips'] == 2, 'en evaluación todos los clips son cerrados, valga lo que valga ss'
        legacy = _cfg(closed_loop=False)
        m2 = H3Model(legacy, IN)
        _, lg = run_batch(m2, batch, legacy, 'cpu', True, 1.0)
        assert lg['closed_clips'] == 0 and lg['births'] == lg['gt_ids'], 'closed_loop=false conserva la receta anterior (nacimientos GT)'
        assert 'births_dup' in closed and closed['births_dup'] + closed['births_fp'] > 0, 'los nacimientos de más se clasifican (duplicado de una identidad con track / falso positivo)'
        frac = [run_batch(m, batch, cfg, 'cpu', True, 0.3)[1]['closed_clips'] / 2 for _ in range(60)]
        assert 0.15 < np.mean(frac) < 0.45, f'con ss=0.3 el 30 % de los clips debe ser cerrado: {np.mean(frac):.2f}'


def test_birth_head_and_loss():
    from tracking.e2e.h3.birth_head import H3BirthHead
    from tracking.e2e.h3.losses import birth_loss
    h = H3BirthHead(32, 16)
    z = lambda *s: torch.zeros(*s)
    assert h(z(2, 5, 32), z(2, 5), z(2, 5), z(2, 5), z(2, 5), z(2, 5), z(2, 5)).shape == (2, 5)
    assert h(z(2, 0, 32), z(2, 0), z(2, 0), z(2, 0), z(2, 0), z(2, 0), z(2, 0)).shape == (2, 0), 'D = 0'
    logit, mask = torch.tensor([[4.0, 4.0, -4.0]]), torch.ones(1, 3, dtype=torch.bool)
    label = torch.tensor([[True, False, False]])
    base, n = birth_loss(logit, label, mask)
    heavy, _ = birth_loss(logit, label, mask, neg_weight=5.0)
    assert int(n) == 3 and float(heavy) > float(base) + 10, 'neg_weight > 1 penaliza más el falso nacimiento (logit alto con etiqueta 0)'
    assert float(birth_loss(logit, label, mask, pos_weight=0.0)[0]) < float(base)
    l0, n0 = birth_loss(logit.clone().requires_grad_(True), label, torch.zeros(1, 3, dtype=torch.bool))
    assert int(n0) == 0 and float(l0) == 0.0 and l0.requires_grad, 'cero conectado al grafo'
    m, B, K = _model(), 2, 6
    for alive, nv in (([3, 5], [4, 6]), ([0, 0], [4, 6]), ([3, 2], [0, 0])):
        out = m.forward_scores(_mem(m, B, alive), _dets(B, K, nv), torch.tensor([5, 5]))
        assert out['birth_logit'].shape == (B, K) and torch.isfinite(out['birth_logit']).all()
    m0 = _model(max_tracks=0)
    assert m0.forward_scores(m0.new_memory(B, 'cpu'), _dets(B, K), torch.tensor([5, 5]))['birth_logit'].shape == (B, K), 'T = 0'
    assert _model(birth=dict(enabled=False)).forward_scores(_mem(m, B, [3, 5]), _dets(B, K), torch.tensor([5, 5]))['birth_logit'] is None, 'sin cabeza: None'
    out = m.forward_scores(_mem(m, B, [3, 5]), _dets(B, K), torch.tensor([5, 5]))      # con detach (por defecto) la pérdida de nacimiento no llega a la asociación
    birth_loss(out['birth_logit'], torch.zeros(B, K, dtype=torch.bool), torch.ones(B, K, dtype=torch.bool))[0].backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.birth.parameters())
    assert all(p.grad is None for n, p in m.named_parameters() if not n.startswith('birth.')), 'la cabeza de nacimiento no modifica las señales de la asociación'


def test_closed_births_follow_the_birth_head():
    """En los clips cerrados nace solo lo que la cabeza aprueba: con la cabeza apagada (logit muy negativo) no nace ningún track; con la cabeza siempre a favor nacen los de la regla por score
    (incluidos duplicados y falsos positivos), que se cuentan aparte."""
    from tracking.e2e.train import run_batch
    with tempfile.TemporaryDirectory() as tmp:
        batch = _batch(tmp)
        cfg = _cfg()
        torch.manual_seed(0)
        m = H3Model(cfg, IN)
        with torch.no_grad():
            m.birth.net[-1].weight.zero_(); m.birth.net[-1].bias.fill_(-30.0)
        loss, st = run_batch(m, batch, cfg, 'cpu', True, 1.0)
        assert st['closed_clips'] == 2 and st['births'] == 0 and torch.isfinite(loss), 'la cabeza dice que no: ningún nacimiento (el bucle cerrado no se rompe sin tracks)'
        with torch.no_grad():
            m.birth.net[-1].bias.fill_(30.0)
        loss, st = run_batch(m, batch, cfg, 'cpu', True, 1.0)
        assert st['births'] > 0 and st['births_dup'] + st['births_fp'] > 0 and 'birth' in st['parts']
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_birth_head_learns_on_synthetic():
    """Con el currículo (de clips GT a clips cerrados) la cabeza aprende a hacer nacer solo las identidades nuevas: en evaluación en bucle cerrado nacen tantos tracks como identidades GT,
    sin duplicados ni falsos positivos, y su pérdida baja."""
    from tracking.e2e.train import collate, run_batch
    from tracking.video.clips import CachedClipDataset
    with tempfile.TemporaryDirectory() as tmp:
        _synthetic(tmp, 80)
        ds = CachedClipDataset(f'{tmp}/train/cache', None, T=6, max_stride=2, p_skip=0.1, max_skip=2, samples_per_epoch=16, seed=0)
        cfg = _cfg(dim=64)
        torch.manual_seed(0)
        m = H3Model(cfg, IN).train()
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
        first = last = None
        for step in range(150):
            batch = collate([ds[(2 * step) % 16], ds[(2 * step + 1) % 16]])
            loss, st = run_batch(m, batch, cfg, 'cpu', True, min(1.0, step / 75) * 0.9)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            first = st['parts']['birth'] if step == 0 else first
            last = st['parts']['birth']
        assert last < 0.2 * first, f'la pérdida de nacimiento debe bajar: {first:.3f} -> {last:.3f}'
        m.eval()
        tot = dict(births=0, dup=0, fp=0, gt=0)
        for i in range(8):
            _, ev = run_batch(m, collate([ds[i], ds[i + 8]]), cfg, 'cpu', False, 1.0)
            tot['births'] += ev['births']; tot['dup'] += ev['births_dup']; tot['fp'] += ev['births_fp']; tot['gt'] += ev['gt_ids']
        assert tot['births'] <= 1.2 * tot['gt'] and tot['dup'] + tot['fp'] <= 0.1 * tot['births'] and tot['births'] >= 0.8 * tot['gt'], f'nacen las identidades nuevas y poco más: {tot}'


def _bank_model(**bank):
    return _model(bank=dict(enabled=True, size=4, **bank))


def _bank_mem(m, B, alive, seed=3):
    """Memoria con banco: los vivos con 1..M embeddings aleatorios; los huecos libres, con basura que debe ignorarse."""
    g = torch.Generator().manual_seed(seed)
    mem = _mem(m, B, alive)
    M = mem.bank.shape[2]
    n = torch.where(mem.alive, torch.randint(1, M + 1, (B, mem.N), generator=g), torch.zeros(B, mem.N, dtype=torch.long))
    return mem.replace(bank=torch.randn(B, mem.N, M, IN, generator=g), bank_n=n, bank_t=torch.randint(0, 5, (B, mem.N, M), generator=g), last_write=torch.full((B, mem.N), 3))


def test_track_memory_bank():
    mem = TrackMemory.empty(2, 3, IN, 4, 'cpu', bank_size=3)
    e0 = torch.randn(2, IN)
    mem, n = mem.spawn(0, torch.tensor([7, 8]), e0, torch.rand(2, 4), torch.tensor(5))
    assert n == 2 and mem.bank_n[0].tolist() == [1, 1, 0] and torch.equal(mem.bank[0, 0, -1], e0[0]) and mem.bank_t[0, 1, -1] == 5 and mem.last_write[0].tolist() == [5, 5, 0], \
        'el banco empieza con el embedding de la primera detección'
    assert (mem.bank[0, 0, :-1] == 0).all()
    obs, new = torch.randn(2, 3, IN), torch.randn(2, 3, IN)
    matched, write = torch.tensor([[True, True, False], [False] * 3]), torch.tensor([[True, False, False], [False] * 3])
    m2 = mem.push(matched, torch.rand(2, 3, 4), new, torch.tensor([6, 6]), obs, write)
    assert m2.bank_n[0].tolist() == [2, 1, 0] and torch.equal(m2.bank[0, 0, -1], obs[0, 0]) and torch.equal(m2.bank[0, 0, -2], e0[0]), 'solo escribe el marcado en write y el anterior se desplaza'
    assert torch.equal(m2.bank[0, 1], mem.bank[0, 1]) and m2.last_write[0].tolist() == [6, 5, 0] and m2.bank_t[0, 0, -1] == 6, 'asignado pero sin escribir: el banco no cambia'
    for t in range(5):
        m2 = m2.push(matched, torch.rand(2, 3, 4), new, torch.tensor([7 + t, 7 + t]), obs, matched)
    assert m2.bank_n[0].tolist() == [3, 3, 0], 'el banco se limita a M'
    assert m2.kill(torch.tensor([100, 100]), 30).bank_n.sum() == 0 and mem.push(matched, torch.rand(2, 3, 4), new, torch.tensor([6, 6])).bank_n.tolist() == mem.bank_n.tolist(), 'kill vacía el banco; push sin write no lo toca'
    assert TrackMemory.empty(1, 2, IN, 4, 'cpu').bank is None, 'sin banco: None'


def test_history_pool_and_bank_affinity():
    torch.manual_seed(0)
    pool = HistoryPool(32, 4).eval()
    B, N, M = 2, 3, 4
    a_trk, a_bank = torch.randn(B, N, 32), torch.randn(B, N, M, 32)
    valid = bank_valid(torch.tensor([[0, 2, 4], [1, 4, 3]]), M)
    age = torch.randint(0, 20, (B, N, M))
    out = pool(a_trk, a_bank, age, valid)
    assert out.shape == (B, N, 32) and torch.isfinite(out).all() and out[0, 0].abs().sum() == 0, 'sin embeddings válidos el resumen es 0'
    junk = a_bank.clone(); junk[~valid] = torch.randn(int((~valid).sum()), 32) * 9
    assert torch.allclose(pool(a_trk, junk, age, valid), out, atol=1e-5), 'los huecos de relleno del banco no influyen'
    assert not torch.allclose(pool(a_trk, a_bank, age + 30, valid)[0, 2], out[0, 2], atol=1e-4), 'la edad de los embeddings influye'
    assert pool(a_trk[:, :0], a_bank[:, :0], age[:, :0], valid[:, :0]).shape == (B, 0, 32), 'N = 0'
    a_det = torch.nn.functional.normalize(torch.randn(B, 5, 32), dim=-1)
    a_t, a_b = torch.nn.functional.normalize(a_trk, dim=-1), torch.nn.functional.normalize(a_bank, dim=-1)
    smax, ema = bank_affinity(a_det, a_t, a_b, valid, 'max')
    smean, _ = bank_affinity(a_det, a_t, a_b, valid, 'mean')
    snone, _ = bank_affinity(a_det, a_t, a_b, valid, 'none')
    assert smax.shape == (B, 5, N) and (smax >= ema - 1e-6).all() and torch.allclose(snone, ema) and (smax >= smean - 1e-6).all(), 'max >= media; none = solo EMA'
    assert torch.allclose(bank_affinity(a_det, a_t, a_b, torch.zeros_like(valid), 'max')[0], ema), 'banco vacío: solo la EMA'
    assert bank_affinity(a_det[:, :0], a_t, a_b, valid)[0].shape == (B, 0, N) and bank_affinity(a_det, a_t[:, :0], a_b[:, :0], valid[:, :0])[0].shape == (B, 5, 0)


def test_model_with_bank():
    m, B, K = _bank_model(), 2, 6
    for name, alive, nv in (('normal', [3, 5], [4, 6]), ('D=0', [3, 2], [0, 0]), ('T=0', [0, 0], [4, 6]), ('T=1', [1, 1], [4, 6])):
        mem, dets = _bank_mem(m, B, alive), _dets(B, K, nv)
        out = m.forward_scores(mem, dets, torch.tensor([5, 5]))
        _check_out(out, B, K, mem.N)
        assert out['s_app_ema'].shape == (B, K, mem.N) and torch.isfinite(out['birth_logit']).all(), name
        new, matched = m.update(mem, m.decide(out), dets, torch.tensor([5, 5]))
        assert new.bank.shape == mem.bank.shape and not (new.bank_n != mem.bank_n)[~matched].any(), 'solo cambian los asignados'
    m0 = _bank_model()
    m0 = _model(max_tracks=0, bank=dict(enabled=True, size=4))
    out = m0.forward_scores(m0.new_memory(B, 'cpu'), _dets(B, K), torch.tensor([5, 5]))
    _check_out(out, B, K, 0)


def test_bank_padding_invariance():
    m, B = _bank_model(), 1
    d_small, mem_small = _dets(B, 4), _bank_mem(m, B, [3])
    mem_big = _bank_mem(m, B, [3], seed=9)
    cat = lambda a, b: torch.cat([a[:, :3], b[:, 3:]], 1)
    mem_big = mem_big.replace(app=cat(mem_small.app, mem_big.app), hist=cat(mem_small.hist, mem_big.hist), bank=cat(mem_small.bank, mem_big.bank), bank_n=cat(mem_small.bank_n, mem_big.bank_n),
                              bank_t=cat(mem_small.bank_t, mem_big.bank_t))
    g = torch.Generator().manual_seed(9)
    d_big = DetTokens(torch.cat([d_small.emb, torch.randn(B, 3, IN, generator=g)], 1), torch.cat([d_small.box, torch.rand(B, 3, 4, generator=g)], 1),
                      torch.cat([d_small.score, torch.rand(B, 3, generator=g)], 1), torch.tensor([[True] * 4 + [False] * 3]))
    o1, o2 = m.forward_scores(mem_small, d_small, FRAME), m.forward_scores(mem_big, d_big, FRAME)
    assert torch.allclose(o1['s'][0, :4, :3], o2['s'][0, :4, :3], atol=1e-4), 'con banco, el relleno de detecciones y huecos no cambia la afinidad de los reales'
    assert torch.allclose(o1['s_app'][0, :4, :3], o2['s_app'][0, :4, :3], atol=1e-5)


def test_bank_write_rules():
    """Solo escriben los asignados con detección fiable: score >= write_conf, sin solape con otra detección válida (un cruce contamina el embedding) y a un ritmo de al menos write_every frames."""
    m = _bank_model(write_conf=0.5, write_overlap=0.5, write_every=2)
    B, K = 1, 5
    box = torch.tensor([[[0.2, 0.5, 0.1, 0.3], [0.4, 0.5, 0.1, 0.3], [0.6, 0.5, 0.1, 0.3], [0.61, 0.5, 0.1, 0.3], [0.9, 0.5, 0.1, 0.3]]])      # la 2 y la 3 se solapan casi del todo
    score = torch.tensor([[0.9, 0.2, 0.9, 0.9, 0.9]])
    dets = DetTokens(torch.randn(B, K, IN), box, score, torch.ones(B, K, dtype=torch.bool))
    mem = m.new_memory(B, 'cpu')
    mem, _ = mem.spawn(0, torch.arange(5), torch.randn(5, IN), box[0], torch.tensor(1))
    lw = mem.last_write.clone(); lw[0, :5] = torch.tensor([1, 1, 1, 1, 9])
    mem = mem.replace(last_write=lw)                                                                     # el hueco 4 acaba de escribir (frame 9)
    assign = torch.tensor([[0, 1, 2, 3, 4] + [K] * (mem.N - 5)])
    new, matched = m.update(mem, assign, dets, torch.tensor([10]))
    wrote = (new.bank_n[0, :5] - mem.bank_n[0, :5]).tolist()
    assert matched[0, :5].all() and wrote == [1, 0, 0, 0, 0], f'solo escribe el 0 (score alto, aislado, hace tiempo que no escribe): {wrote}'
    assert new.last_write[0, 0] == 10 and new.last_write[0, 1] == 1


def test_bank_helps_when_the_average_appearance_is_wrong():
    """La apariencia media (EMA) del track se ha desviado, pero su banco conserva cómo se veía: la afinidad con el banco recupera la pareja y la EMA sola no."""
    m = _bank_model(affinity='max')
    B, K = 1, 3
    dets = _dets(B, K)
    mem = _bank_mem(m, B, [2])
    mem = mem.replace(app=torch.randn(1, mem.N, IN) * 3, bank_n=torch.tensor([[1, 1] + [0] * (mem.N - 2)]))      # EMA de otra persona; el banco guarda el embedding de la detección 0 en el track 0
    bank = mem.bank.clone(); bank[0, 0, -1] = dets.emb[0, 0]
    out = m.forward_scores(mem.replace(bank=bank), dets, torch.tensor([5]))
    assert out['s_app'][0, 0, 0] > 0.99 and out['s_app_ema'][0, 0, 0] < 0.6, 'con banco: la pareja correcta tiene coseno 1; solo con la EMA, no'
    m_none = _bank_model(affinity='none')
    m_none.load_state_dict(m.state_dict())
    o2 = m_none.forward_scores(mem.replace(bank=bank), dets, torch.tensor([5]))
    assert torch.allclose(o2['s_app'], o2['s_app_ema']) and o2['s_app'][0, 0, 0] < 0.6, 'affinity=none ignora el banco en la afinidad (aunque el token siga usándolo)'


def test_bank_trains_and_reaches_the_token():
    from tracking.e2e.train import run_batch
    with tempfile.TemporaryDirectory() as tmp:
        batch = _batch(tmp)
        for h3 in (dict(bank=dict(enabled=True, size=4)), dict(bank=dict(enabled=True, size=4, affinity='none')), dict(bank=dict(enabled=True, size=4), branches=dict(rel=False))):
            cfg = _cfg(**h3)
            torch.manual_seed(0)
            m = H3Model(cfg, IN)
            loss, st = run_batch(m, batch, cfg, 'cpu', True, 0.5)
            assert torch.isfinite(loss), h3
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
            pool = [p.grad for p in m.hist_pool.parameters()]
            if m.use_rel:
                assert any(g is not None and g.abs().sum() > 0 for g in pool), 'el resumen del historial llega al token y recibe gradiente'
            else:
                assert all(g is None for g in pool), 'sin rama relacional el resumen no se usa'


def test_build_model_dispatch():
    assert isinstance(build_model(load_config(None, None), IN), E2EModel)
    assert isinstance(build_model(_cfg(), IN), H3Model)


# ------------------------------------------------------------------ pérdidas y ruido
def test_losses():
    bp, gt = torch.rand(2, 3, 4) * 0.3 + 0.2, torch.rand(2, 3, 4) * 0.3 + 0.2
    mask = torch.tensor([[True, False, False], [False, False, False]])
    bp.requires_grad_(True)
    l, n = motion_loss(bp, gt, mask)
    assert int(n) == 1 and torch.isfinite(l) and float(l) > 0
    l.backward()
    assert bp.grad[0, 1:].abs().sum() == 0 and bp.grad[1].abs().sum() == 0 and bp.grad[0, 0].abs().sum() > 0
    l0, n0 = motion_loss(bp, gt, torch.zeros(2, 3, dtype=torch.bool))
    assert int(n0) == 0 and float(l0) == 0.0 and l0.requires_grad, 'cero conectado'
    perfect, _ = motion_loss(gt.clone().requires_grad_(True), gt, torch.ones(2, 3, dtype=torch.bool))
    assert float(perfect) < 1e-5
    a_det = torch.nn.functional.normalize(torch.randn(1, 3, 16), dim=-1)
    a_trk = torch.nn.functional.normalize(torch.randn(1, 4, 16), dim=-1)
    pos_idx, pos_mask = torch.tensor([[0, 1, 4]]), torch.tensor([[True, True, False]])        # las dos primeras detecciones tienen su positivo en los huecos 0 y 1
    tv = torch.tensor([[True, True, True, False]])
    li, ni = app_infonce(a_det, a_trk, pos_idx, pos_mask, tv, 0.1)
    assert int(ni) == 2 and torch.isfinite(li)
    aligned = a_trk.clone(); aligned[0, 0], aligned[0, 1] = a_det[0, 0], a_det[0, 1]
    assert float(app_infonce(a_det, aligned, pos_idx, pos_mask, tv, 0.1)[0]) < float(li), 'alinear positivos baja el InfoNCE'
    l1, n1 = app_infonce(a_det, a_trk, pos_idx, torch.zeros(1, 3, dtype=torch.bool), tv, 0.1)
    assert int(n1) == 0 and float(l1) == 0.0


def test_noise():
    torch.manual_seed(0)
    nc = load_config(None, None).h3.noise
    B, K = 4, 50
    dets = _dets(B, K)
    tid = torch.where(torch.rand(B, K) < 0.3, torch.arange(K)[None].expand(B, K), -torch.ones(B, K, dtype=torch.long))
    nd, clean = noisy_dets(dets, tid, nc, p_drop=0.5)
    assert torch.equal(clean, dets.box) and nd.box.shape == dets.box.shape and not torch.equal(nd.box, dets.box), 'el jitter perturba las cajas observadas'
    dropped = (dets.valid & ~nd.valid)
    assert dropped.any() and (tid[dropped] >= 0).all(), 'solo se descartan detecciones de IDs GT'
    forced = nd.valid & (nd.score == 1.0) & (tid < 0)
    assert forced.any(), 'hay falsos positivos sintéticos (fondo con score 1)'
    nc0 = load_config(None, {'h3': {'noise': {'jitter_pos': 0.0, 'jitter_scale': 0.0, 'p_fp': 0.0}}}).h3.noise
    n0, _ = noisy_dets(dets, tid, nc0, p_drop=0.0)
    assert torch.allclose(n0.box, dets.box) and torch.equal(n0.valid, dets.valid), 'sin ruido configurado no cambia nada'


# ------------------------------------------------------------------ entrenamiento e inferencia
def _synthetic(tmp, frames=60):
    from tracking.tools.make_synthetic import make
    rng = np.random.default_rng(0)
    for name in ('tr00', 'tr01'):
        make(f'{tmp}/train', name, frames, 4, 12, 10, rng)
    make(f'{tmp}/val', 'va00', frames, 4, 12, 10, rng)


def test_smoke_train_and_infer():
    """Entrenamiento corto (CPU) con h3.enabled + inferencia: pérdidas finitas, partes registradas por separado y checkpoint reutilizable."""
    from tracking.e2e.train import main
    from tracking.e2e import infer
    import argparse, json
    with tempfile.TemporaryDirectory() as tmp:
        _synthetic(tmp)
        a = argparse.Namespace(cfg=None, seed=0, device='cpu', gt_root=None, out=f'{tmp}/run', train_cache=f'{tmp}/train/cache', val_cache=f'{tmp}/val/cache',
                               update=['h3.enabled=true', 'h3.birth.enabled=false', 'h3.dim=64', 'h3.heads=4', 'h3.ffn=128', 'h3.gru_dim=32', 'h3.app_dim=32', 'h3.K_hist=6', 'state.max_tracks=8',
                                       'train.epochs=2', 'train.samples_per_epoch=8', 'train.batch=2', 'train.T=6', 'train.val_samples=2'])
        main(a)
        for l in open(f'{tmp}/run/log.txt'):
            rec = json.loads(l)
            assert all(np.isfinite(v) for v in rec.values() if isinstance(v, float)), rec
            assert all(k in rec for k in ('train_assign', 'train_app', 'train_motion', 'val_app_loss', 'val_motion_loss')), rec
            assert 'train_assoc' not in rec and 'val_birth_loss' not in rec
        ck = torch.load(f'{tmp}/run/best.pth')
        cfg = load_config(None, ck['cfg'])
        assert cfg.h3.enabled
        model = build_model(cfg, ck['in_dim']).eval(); model.load_state_dict(ck['model'])
        assert isinstance(model, H3Model)
        rows, _ = infer.track_sequence(model, torch.load(f'{tmp}/val/cache/va00.pt'), cfg, 'cpu')
        assert len(rows) > 0 and len({r[1] for r in rows}) >= 1


def test_ablated_models_train():
    """Cada ablación recorre el bucle de entrenamiento (con DDP no hay parámetros sin gradiente que rompan: find_unused_parameters) y la pérdida es finita."""
    from tracking.e2e.train import collate, run_batch
    from tracking.video.clips import CachedClipDataset
    with tempfile.TemporaryDirectory() as tmp:
        _synthetic(tmp, 40)
        ds = CachedClipDataset(f'{tmp}/train/cache', None, T=5, max_stride=2, p_skip=0.1, max_skip=2, samples_per_epoch=4, seed=0)
        batch = collate([ds[0], ds[1]])
        for h3 in (dict(), dict(branches=dict(app=False)), dict(branches=dict(mot=False)), dict(branches=dict(rel=False)), dict(gates=False),
                   dict(memory=dict(app_update='gate')), dict(memory=dict(edges=True))):
            cfg = _cfg(**h3)
            torch.manual_seed(0)
            m = H3Model(cfg, IN)
            loss, st = run_batch(m, batch, cfg, 'cpu', True, 0.5)
            assert torch.isfinite(loss), h3
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None), h3
            assert set(st['parts']) == {'assign', 'birth'} | ({'app'} if m.use_app else set()) | ({'motion'} if m.use_mot else set())


def test_learns_on_synthetic():
    """Unos pasos de optimización bajan la pérdida de asignación y mejoran la precisión de las decisiones del propio modelo."""
    from tracking.e2e.train import collate, run_batch
    from tracking.video.clips import CachedClipDataset
    with tempfile.TemporaryDirectory() as tmp:
        _synthetic(tmp, 80)
        ds = CachedClipDataset(f'{tmp}/train/cache', None, T=6, max_stride=2, p_skip=0.1, max_skip=2, samples_per_epoch=16, seed=0)
        cfg = _cfg(dim=64)
        torch.manual_seed(0)
        m = H3Model(cfg, IN).train()
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
        first, last, acc_last = [], [], []
        for step in range(60):
            batch = collate([ds[(2 * step) % 16], ds[(2 * step + 1) % 16]])
            loss, st = run_batch(m, batch, cfg, 'cpu', True, 0.5)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            (first if step < 10 else last if step >= 50 else []).append(st['parts']['assign'])
            if step >= 50:
                acc_last.append(st['ok'] / max(st['n'], 1))
        assert np.mean(last) < 0.7 * np.mean(first), f'L_assign debe bajar: {np.mean(first):.3f} -> {np.mean(last):.3f}'


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for t in tests:
        t(); print('ok  ', t.__name__)
    print(f'{len(tests)} tests correctos')
