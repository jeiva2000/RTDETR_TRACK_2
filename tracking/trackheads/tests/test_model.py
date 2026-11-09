"""Tests de TrackHeads (etapa 1): formas, centros de los tokens, inicialización (caja = prev_box, presencia 0.5, ΔW = 0), prior gaussiano,
independencia entre tracks (relleno), equivalencia de la corrección de rango r con ΔW_i explícita, decodificación de la caja, gradientes
y un test de aprendizaje: localizar un objeto desplazado en una memoria sintética.
Ejecución: python -m tracking.trackheads.tests.test_model   (o pytest tracking/trackheads/tests)"""
import math, os, sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from tracking.trackheads.model import TrackHeads, sine_2d, token_centers

D = 256
SHAPES = [(16, 16), (8, 8), (4, 4)]           # strides 8/16/32 de una imagen 128×128
N = sum(h * w for h, w in SHAPES)


def _inputs(B=2, T=5, seed=0, n_valid=None):
    g = torch.Generator().manual_seed(seed)
    emb = torch.randn(B, T, D, generator=g)
    box = torch.cat([torch.rand(B, T, 2, generator=g) * 0.6 + 0.2, torch.rand(B, T, 2, generator=g) * 0.2 + 0.05], -1)
    feats = [torch.randn(B, D, h, w, generator=g) for h, w in SHAPES]
    mask = torch.ones(B, T, dtype=torch.bool) if n_valid is None else torch.arange(T)[None] < torch.tensor(n_valid)[:, None]
    return emb, box, feats, mask


def _model(**kw):
    torch.manual_seed(0)
    return TrackHeads(**kw)


def test_shapes():
    m = _model()
    emb, box, feats, mask = _inputs(n_valid=[5, 2])
    o = m(emb, box, feats, mask)
    assert o['boxes'].shape == (2, 5, 4) and o['presence_logits'].shape == (2, 5)
    assert o['attn'].shape == (2, 5, N) and o['token_xy'].shape == (N, 2)
    assert torch.equal(o['track_mask'], mask)
    assert torch.allclose(o['attn'].sum(-1), torch.ones(2, 5), atol=1e-5)
    assert all(torch.isfinite(v).all() for v in (o['boxes'], o['presence_logits'], o['attn']))


def test_token_centers_follow_flatten_order():
    xy, lvl = token_centers([(2, 3), (1, 1)])
    assert xy.shape == (7, 2) and lvl.tolist() == [0] * 6 + [1]
    # fila a fila: (x varía primero)
    assert torch.allclose(xy[:3, 1], torch.full((3,), 0.25)) and torch.allclose(xy[:3, 0], torch.tensor([1 / 6, 0.5, 5 / 6]))
    assert torch.allclose(xy[3, :], torch.tensor([1 / 6, 0.75])) and torch.allclose(xy[6], torch.tensor([0.5, 0.5]))
    # coincide con el orden de feat.flatten(2): el valor del canal 0 en (fila i, col j) es 10i + j
    f = (torch.arange(2)[:, None] * 10 + torch.arange(3)[None]).float()[None, None]
    flat = f.flatten(2)[0, 0]
    assert torch.allclose(flat, (xy[:6, 1] * 2 - 0.5) * 10 + (xy[:6, 0] * 3 - 0.5))


def test_sine_2d():
    xy = torch.rand(10, 2)
    p = sine_2d(xy, 256)
    assert p.shape == (10, 256) and p.abs().max() <= 1 + 1e-6
    assert not torch.allclose(sine_2d(torch.tensor([[0.1, 0.2]]), 256), sine_2d(torch.tensor([[0.2, 0.1]]), 256))


def test_init_identity():
    """Última capa a cero: caja = prev_box, presencia 0.5; V a cero: la corrección ΔW_i no cambia nada al inicio."""
    m = _model()
    emb, box, feats, mask = _inputs()
    o = m(emb, box, feats, mask)
    assert torch.allclose(o['boxes'], box, atol=1e-6)
    assert torch.allclose(o['presence_logits'], torch.zeros(2, 5))
    _, _, V = m.gen(emb, box)
    assert V.abs().max() == 0


def test_prior_concentrates_attention():
    """Con el prior y una caja pequeña, la atención se concentra cerca de c_i; sin el prior, no."""
    emb, box, feats, mask = _inputs()
    box[..., 2:] = 0.03
    for prior, expect_close in ((True, True), (False, False)):
        m = _model(use_prior=prior)
        o = m(emb, box, feats, mask)
        center = o['attn'] @ o['token_xy']
        err = (center - box[..., :2]).norm(dim=-1).max().item()
        assert (err < 0.05) == expect_close, (prior, err)


def test_scale_floor_and_init():
    m = _model()
    assert abs(m.log_scale.item() - math.log(2)) < 1e-6
    s = m.scale(torch.tensor([[[0.5, 0.5, 0.001, 0.002], [0.5, 0.5, 0.1, 0.3]]]))
    assert torch.allclose(s, torch.tensor([[0.02, 0.6]]))


def test_tracks_are_independent():
    """Cada cabeza mira solo la memoria: la salida de un track no depende de los demás (ni del relleno) y permutar tracks permuta salidas."""
    m = _model()
    _perturb(m)
    emb, box, feats, mask = _inputs(B=1, T=6)
    o = m(emb, box, feats, mask)
    o3 = m(emb[:, :3], box[:, :3], feats, mask[:, :3])
    assert torch.allclose(o['boxes'][:, :3], o3['boxes'], atol=1e-5)
    assert torch.allclose(o['attn'][:, :3], o3['attn'], atol=1e-6)
    p = torch.randperm(6)
    op = m(emb[:, p], box[:, p], feats, mask[:, p])
    assert torch.allclose(op['boxes'], o['boxes'][:, p], atol=1e-5)
    assert torch.allclose(op['presence_logits'], o['presence_logits'][:, p], atol=1e-5)


def _perturb(m):
    """Pesos no nulos en V y en la salida para que la corrección y la cabeza final cuenten en los tests."""
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in (m.gen.v.weight, m.gen.v.bias, m.out[-1].weight, m.out[-1].bias):
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)


def test_low_rank_matches_explicit_delta_w():
    """v = W_v x̄ + (x̄ U_i) V_iᵀ debe ser igual a promediar por token con la proyección propia (W_v + U_i V_iᵀ) (lo que no se materializa)."""
    m = _model()
    _perturb(m)
    emb, box, feats, mask = _inputs(B=1, T=3)
    mem, _, _ = m.memory(feats)
    o = m(emb, box, feats, mask)
    _, U, V = m.gen(emb, box)
    dW = U @ V.transpose(-1, -2)                                                     # (B,T,D,d_h)
    per_token = m.w_v(mem)[:, None] + torch.einsum('bnd,btdh->btnh', mem, dW)        # (B,T,N,d_h): solo en el test
    v_ref = (o['attn'][..., None] * per_token).sum(2)
    xbar = o['attn'] @ mem
    v = m.w_v(xbar) + torch.einsum('btr,bthr->bth', torch.einsum('btd,btdr->btr', xbar, U), V)
    assert torch.allclose(v, v_ref, atol=1e-4)


def test_decode():
    m = _model()
    prev = torch.tensor([[[0.5, 0.4, 0.2, 0.1]]])
    out = m.decode(prev, torch.tensor([[[0.5, -1.0, math.log(2), 10.0]]]))
    assert torch.allclose(out, torch.tensor([[[0.6, 0.3, 0.4, 0.1 * math.exp(3)]]]), atol=1e-6)


def test_gradients_reach_every_part():
    """Llegan gradientes a la hipernetwork, W_k, W_v, posición, nivel y escala. Por las dos inicializaciones a cero el gradiente se abre
    en cascada: paso 1 solo la última capa de salida; paso 2 ya V y el resto; U recibe gradiente desde el paso 3 (cuando V ≠ 0)."""
    m = _model()
    emb, box, feats, mask = _inputs(n_valid=[5, 3])
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    for _ in range(3):
        o = m(emb, box, feats, mask)
        target = box + 0.05
        loss = (F.l1_loss(o['boxes'], target, reduction='none').sum(-1)[mask]).mean() + \
            F.binary_cross_entropy_with_logits(o['presence_logits'][mask], torch.ones(int(mask.sum())))
        opt.zero_grad()
        loss.backward()
        opt.step()
    for name in ('gen.q.0.weight', 'gen.u.weight', 'gen.v.weight', 'w_k.weight', 'w_v.weight', 'pos_proj.weight', 'level_embed', 'log_scale'):
        p = dict(m.named_parameters())[name]
        assert p.grad is not None and p.grad.abs().sum() > 0, name
    assert m.gen.v.weight.abs().sum() > 0


def _synthetic(B, T, g):
    """Memoria con un "objeto" por track: un patrón propio (la misma dirección que su query) pintado en el nivel fino en la posición
    nueva. La caja anterior está desplazada ±2 tokens respecto a la nueva: la cabeza debe encontrar su patrón y corregir la caja."""
    H, W = SHAPES[0]
    codes = F.normalize(torch.randn(B, T, D, generator=g), dim=-1)
    new_ij = torch.stack([torch.randint(3, H - 3, (B, T), generator=g), torch.randint(3, W - 3, (B, T), generator=g)], -1)
    shift = torch.randint(-2, 3, (B, T, 2), generator=g)
    old_ij = new_ij + shift
    feats = [torch.randn(B, D, h, w, generator=g) * 0.1 for h, w in SHAPES]
    for b in range(B):
        for t in range(T):
            i, j = new_ij[b, t]
            feats[0][b, :, i, j] += codes[b, t] * 4
    to_xy = lambda ij: torch.stack([(ij[..., 1] + 0.5) / W, (ij[..., 0] + 0.5) / H], -1)
    wh = torch.full((B, T, 2), 0.15)
    prev = torch.cat([to_xy(old_ij), wh], -1)
    target = torch.cat([to_xy(new_ij), wh], -1)
    return codes, prev, feats, target


def _target_token(target):
    H, W = SHAPES[0]
    return ((target[..., 1] * H - 0.5).round().long()) * W + (target[..., 0] * W - 0.5).round().long()


def test_attention_learns_to_find_object():
    """La cabeza de cada track aprende a encontrar SU patrón (por apariencia, generalizando a códigos nuevos) en la memoria del frame t+k.
    Se supervisa la atención directamente (−log attn en el token del objeto). Nota: con solo la L1 de la caja, en esta tarea sintética la
    atención no aprende en 2000 pasos (la señal que le llega por shift y v es débil); ver docs/TRACKHEADS.md."""
    torch.manual_seed(0)
    m = TrackHeads(hidden=128, gen_hidden=128)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(3)
    for _ in range(600):
        emb, prev, feats, target = _synthetic(4, 4, g)
        o = m(emb * 4, prev, feats, torch.ones(4, 4, dtype=torch.bool))
        loss = -(o['attn'].gather(-1, _target_token(target)[..., None]) + 1e-9).log().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    emb, prev, feats, target = _synthetic(8, 4, torch.Generator().manual_seed(99))
    with torch.no_grad():
        o = m(emb * 4, prev, feats, torch.ones(8, 4, dtype=torch.bool))
    p = o['attn'].gather(-1, _target_token(target)[..., None]).mean().item()
    assert p > 0.3, p                                                                # al inicio ≈ 0.007


def test_overfits_fixed_batch():
    """La cabeza de salida (caja + presencia) ajusta un lote fijo: la pérdida cae a menos de un 30 % de la inicial."""
    torch.manual_seed(0)
    m = TrackHeads(hidden=128, gen_hidden=128)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    emb, prev, feats, target = _synthetic(2, 4, torch.Generator().manual_seed(5))
    present = torch.tensor([[1., 1., 0., 1.], [0., 1., 1., 1.]])
    mask = torch.ones(2, 4, dtype=torch.bool)
    losses = []
    for _ in range(200):
        o = m(emb, prev, feats, mask)
        loss = F.l1_loss(o['boxes'], target) * 10 + F.binary_cross_entropy_with_logits(o['presence_logits'], present)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.3 * losses[0], (losses[0], losses[-1])


if __name__ == '__main__':
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        fn()
        print('ok', name)
    print(f'{len(tests)} tests OK')
