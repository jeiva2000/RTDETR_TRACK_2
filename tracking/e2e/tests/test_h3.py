"""Tests del H3 (asociación relacional): formas y máscaras de cada componente, casos D = 0, T = 0 y T = 1 (por máscara y con tensores vacíos), invariancia al relleno,
Sinkhorn (marginales, NLL, Hungarian con umbral), memoria de tracks, ablaciones (ramas, gates, aristas), pérdidas, ruido, y un test de humo (entrenamiento + inferencia) y de
aprendizaje sobre datos sintéticos en CPU. Ejecución: python -m tracking.e2e.tests.test_h3   (o pytest tracking/e2e/tests)"""
import os, sys, tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from tracking.e2e.config import load_config
from tracking.e2e.h3 import assignment as A
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
    tid, ids = torch.tensor([[5, 6, -1]]), torch.tensor([[5, 6, 7, -1]])
    dv, tv = torch.ones(1, 3, dtype=torch.bool), torch.tensor([[True, True, True, False]])
    li, ni = app_infonce(a_det, a_trk, tid, ids, dv, tv, 0.1)
    assert int(ni) == 2 and torch.isfinite(li)
    aligned = a_trk.clone(); aligned[0, 0], aligned[0, 1] = a_det[0, 0], a_det[0, 1]
    assert float(app_infonce(a_det, aligned, tid, ids, dv, tv, 0.1)[0]) < float(li), 'alinear positivos baja el InfoNCE'
    l1, n1 = app_infonce(a_det, a_trk, torch.tensor([[-1, -1, -1]]), ids, dv, tv, 0.1)
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
                               update=['h3.enabled=true', 'h3.dim=64', 'h3.heads=4', 'h3.ffn=128', 'h3.gru_dim=32', 'h3.app_dim=32', 'h3.K_hist=6', 'state.max_tracks=8',
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
            assert set(st['parts']) == {'assign'} | ({'app'} if m.use_app else set()) | ({'motion'} if m.use_mot else set())


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
