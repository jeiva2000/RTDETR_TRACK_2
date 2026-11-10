"""Tests del entrenamiento de TrackHeads sin GPU ni datos reales: un DanceTrack sintético en miniatura (imágenes + gt.txt + seqinfo.ini) y un
detector simulado con la misma interfaz que FrozenDetector.detect. Cubren build_tracks (filtros, presencia, caja objetivo, relleno), las
pérdidas (enmascarado, atención dentro de la caja), el muestreo de pares (val fijo, train reproducible por época) y un humo de main()
(entrena, valida, guarda best/last, escribe el log y reanuda).
Ejecución: python -m tracking.trackheads.tests.test_train   (o pytest tracking/trackheads/tests)"""
import json, os, sys, tempfile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from tracking.trackheads.data import PairDataset, collate
from tracking.trackheads.losses import attn_mass_in_box, batch_stats, summarize, trackheads_loss
from tracking.trackheads.model import TrackHeads
from tracking.trackheads.targets import assoc_targets, build_tracks
from tracking.trackheads.train import main, parser
from tracking.trackheads import diagnose as diag

D = 256


def make_mot_root(root, seqs=(('train1', 'tr0', 30), ('train1', 'tr1', 25), ('val', 'va0', 20)), W=96, H=64):
    """Secuencias con 3 personas (rectángulos de colores) que se mueven; la persona 3 sale de escena en los frames 8-12 (presencia 0)."""
    rng = np.random.default_rng(0)
    for split, name, n in seqs:
        sd = os.path.join(root, split, name)
        os.makedirs(os.path.join(sd, 'img1')); os.makedirs(os.path.join(sd, 'gt'))
        open(os.path.join(sd, 'seqinfo.ini'), 'w').write(
            f'[Sequence]\nname={name}\nimDir=img1\nframeRate=20\nseqLength={n}\nimWidth={W}\nimHeight={H}\nimExt=.jpg\n')
        rows = []
        x0 = rng.uniform(5, 60, 3); vx = rng.uniform(-1, 1, 3)
        for f in range(1, n + 1):
            img = np.full((H, W, 3), 40, np.uint8)
            for i in range(3):
                if i == 2 and 8 <= f <= 12:
                    continue
                x, y, w, h = x0[i] + vx[i] * f, 10 + 12 * i, 14, 18
                img[int(y):int(y + h), int(max(x, 0)):int(max(x + w, 0))] = [(i + 1) * 70, 255 - i * 80, 100]
                rows.append([f, i + 1, x, y, w, h, 1, 1, 1])
            Image.fromarray(img).save(os.path.join(sd, 'img1', f'{f:08d}.jpg'))
        np.savetxt(os.path.join(sd, 'gt', 'gt.txt'), np.array(rows), fmt='%.2f', delimiter=',')


class StubDetector:
    """Interfaz de FrozenDetector.detect: queries = cajas GT con ruido + 2 falsos positivos; emb = código propio de cada identidad + ruido;
    feats = imagen promediada a strides 8/16/32 y proyectada a 256 canales con pesos fijos."""
    def __init__(self, K=8):
        g = torch.Generator().manual_seed(0)
        self.codes = torch.randn(50, D, generator=g)
        self.proj = torch.randn(D, 3, generator=g)
        self.K = K

    def detect(self, images, gts, min_iou=0.5):
        B = images.shape[0]
        emb, boxes = torch.randn(B, self.K, D) * 0.1, torch.rand(B, self.K, 4) * 0.2 + 0.1
        scores, tid = torch.full((B, self.K), 0.1), torch.full((B, self.K), -1, dtype=torch.long)
        for b, g in enumerate(gts):
            n = min(len(g), self.K - 2)
            ids = g[:n, 0].long()
            emb[b, :n] += self.codes[ids]
            boxes[b, :n] = g[:n, 1:5] + torch.randn(n, 4) * 0.005
            scores[b, :n], tid[b, :n] = 0.9, ids
        feats = [torch.einsum('dc,bchw->bdhw', self.proj, F.avg_pool2d(images, s)) for s in (8, 16, 32)]
        return dict(emb=emb, boxes=boxes, scores=scores), tid, feats


def _args(root, out, **kw):
    a = parser().parse_args(['--out', out, '--root', root, '--device', 'cpu', '--num-workers', '0', '--batch', '2', '--samples', '6',
                             '--val-samples', '4', '--epochs', '2', '--warmup', '2', '--print-every', '0', '--k-max', '4', '--lr', '1e-3'])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_build_tracks():
    q = dict(emb=torch.randn(2, 5, D), boxes=torch.rand(2, 5, 4) * 0.2 + 0.1, scores=torch.tensor([[.9, .8, .2, .9, .7], [.9, .1, .1, .1, .1]]))
    tid = torch.tensor([[3, 4, 5, -1, 6], [-1, 7, -1, -1, -1]])
    gt_tk = [torch.tensor([[4, .5, .5, .1, .2], [3, .3, .3, .1, .1], [9, .1, .1, .1, .1]]), torch.zeros(0, 5)]
    tg = build_tracks(q, tid, gt_tk, det_thr=0.3, max_tracks=10)
    # imagen 0: tracks 3, 4, 6 (5 tiene score bajo, la query 3 es fondo), ordenados por score; imagen 1: ninguno (score bajo)
    assert tg['track_mask'].tolist() == [[True, True, True], [False, False, False]]
    assert tg['track_id'][0].tolist() == [3, 4, 6]
    assert tg['present'][0].tolist() == [1.0, 1.0, 0.0]
    assert torch.allclose(tg['target_box'][0, 0], torch.tensor([.3, .3, .1, .1])) and torch.allclose(tg['target_box'][0, 1], torch.tensor([.5, .5, .1, .2]))
    assert torch.equal(tg['emb'][0, 0], q['emb'][0, 0]) and torch.equal(tg['prev_box'][0, 2], q['boxes'][0, 4])
    assert build_tracks(q, tid, gt_tk, 0.3, max_tracks=2)['track_id'][0].tolist() == [3, 4]
    empty = build_tracks(q, torch.full_like(tid, -1), gt_tk)                         # sin tracks: T = 1 enmascarado
    assert empty['track_mask'].shape == (2, 1) and not empty['track_mask'].any()


def _out_tg(B=2, T=4, seed=0):
    torch.manual_seed(seed)
    m = TrackHeads()
    feats = [torch.randn(B, D, s, s) for s in (8, 4, 2)]
    tg = dict(emb=torch.randn(B, T, D), prev_box=torch.rand(B, T, 4) * 0.2 + 0.3, track_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]).bool(),
              present=torch.tensor([[1., 0., 1., 1.], [1., 1., 0., 0.]]), target_box=torch.rand(B, T, 4) * 0.2 + 0.3)
    return m(tg['emb'], tg['prev_box'], feats, tg['track_mask']), tg


def test_loss_ignores_padding_and_absent_boxes():
    out, tg = _out_tg()
    l0, p0 = trackheads_loss(out, tg)
    tg2 = dict(tg, target_box=tg['target_box'].clone())
    tg2['target_box'][0, 3] = 9.0      # relleno
    tg2['target_box'][0, 1] = 9.0      # track ausente: su caja no cuenta
    l1, p1 = trackheads_loss(out, tg2)
    assert torch.isfinite(l0) and abs(l0.item() - l1.item()) < 1e-6, (p0, p1)
    tg3 = dict(tg, track_mask=torch.zeros(2, 4, dtype=torch.bool))
    l3, _ = trackheads_loss(out, tg3)
    assert l3.item() == 0.0 and l3.requires_grad
    l0.backward()


def test_assoc_targets_and_loss():
    tid_k = torch.tensor([[5, 3, -1, 9]])
    cand = torch.tensor([[True, True, True, False]])
    # 3 -> candidata 1; 5 -> 0; 9 está pero no es candidata (score bajo) -> ninguna (4); 7 no está -> ninguna; -1 (relleno) -> ninguna
    assert assoc_targets(torch.tensor([[3, 5, 9, 7, -1]]), tid_k, cand).tolist() == [[1, 0, 4, 4, 4]]
    out, tg = _out_tg()
    C = 6
    emb_k, box_k = torch.randn(2, C, D), torch.rand(2, C, 4) * 0.2 + 0.3
    cand = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]]).bool()
    m = TrackHeads()
    torch.manual_seed(0)
    feats = [torch.randn(2, D, s, s) for s in (8, 4, 2)]
    out = m.associate(m(tg['emb'], tg['prev_box'], feats, tg['track_mask']), emb_k, box_k, cand)
    lg = out['assoc_logits']
    assert lg.shape == (2, 4, C + 1) and (lg[1, :, 2:C] < -1e3).all() and torch.isfinite(lg).all()
    tg['assoc_target'] = torch.tensor([[0, 6, 2, 6], [1, 6, 6, 6]])
    l, p = trackheads_loss(out, tg, w_assoc=1.0)
    assert p['assoc'] > 0
    l.backward()
    for name in ('assoc.track.0.weight', 'assoc.cand.0.weight', 'assoc.null.0.weight', 'assoc.w_iou', 'assoc.w_mass', 'assoc.log_app', 'w_k.weight'):
        g = dict(m.named_parameters())[name].grad
        assert g is not None and g.abs().sum() > 0, name                            # w_k: la masa de la atención sobre las candidatas entrena la atención
    # al inicio (caja = prev_box, w_iou = 5) la candidata con mayor IoU con prev_box suele ganar: arranca como un tracker por IoU
    box_k2 = torch.stack([tg['prev_box'][0, 0], torch.tensor([.9, .9, .05, .05])])[None]
    o2 = m.associate(m(tg['emb'][:1, :1], tg['prev_box'][:1, :1], [f[:1] for f in feats], tg['track_mask'][:1, :1]), emb_k[:1, :2], box_k2,
                     torch.ones(1, 2, dtype=torch.bool))
    assert o2['assoc_logits'][0, 0, :2].argmax() == 0


def test_attn_mass_in_box():
    xy = torch.tensor([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]])
    attn = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]])
    mass, n = attn_mass_in_box(attn, xy, torch.tensor([[[0.75, 0.5, 0.2, 1.0]]]))
    assert torch.allclose(mass, torch.tensor([[0.6]])) and n.tolist() == [[2]]


def test_stats_identity_model_matches_prev_box():
    out, tg = _out_tg()
    s = summarize(batch_stats(out, tg))
    assert abs(s['iou'] - s['iou_prev']) < 1e-6 and abs(s['iou_gain']) < 1e-6      # al inicio la caja es prev_box
    assert s['n_tracks'] == 5 and abs(s['present_frac'] - 4 / 5) < 1e-6


def test_pairs_sampling():
    with tempfile.TemporaryDirectory() as d:
        make_mot_root(d)
        va1, va2 = PairDataset(d, 'val', 1, 4, 5, seed=1), PairDataset(d, 'val', 1, 4, 5, seed=1)
        assert va1.pairs == va2.pairs
        va1.set_epoch(3)
        assert [va1[i]['t'] for i in range(5)] == [p[1] for p in va2.pairs]
        tr = PairDataset(d, 'train', 1, 4, 20, seed=1)
        e0 = [(tr[i]['seq'], tr[i]['t'], tr[i]['k']) for i in range(20)]
        assert e0 == [(tr[i]['seq'], tr[i]['t'], tr[i]['k']) for i in reversed(range(20))][::-1]     # independiente del orden
        tr.set_epoch(1)
        assert e0 != [(tr[i]['seq'], tr[i]['t'], tr[i]['k']) for i in range(20)]
        x = tr[0]
        assert x['img_t'].shape == (3, 640, 640) and 1 <= x['k'] <= 4 and x['gt_t'].shape[1] == 5
        b = collate([tr[0], tr[1]])
        assert b['img_tk'].shape == (2, 3, 640, 640) and len(b['gt_tk']) == 2
        # la persona 3 falta en los frames 8-12 del GT
        g =PairDataset(d, 'train', seqs=['tr0']).seqs[0][2]
        assert 3 not in g[10][:, 0].tolist() and 3 in g[5][:, 0].tolist()


def test_train_smoke_and_resume():
    with tempfile.TemporaryDirectory() as d:
        root, out = os.path.join(d, 'data'), os.path.join(d, 'out')
        make_mot_root(root)
        det = StubDetector()
        best = main(_args(root, out), det)
        assert os.path.isfile(os.path.join(out, 'best.pth')) and os.path.isfile(os.path.join(out, 'last.pth'))
        log = [json.loads(l) for l in open(os.path.join(out, 'log.txt'))]
        assert [r['epoch'] for r in log] == [0, 1] and best == max(r['val_assoc_head'] for r in log)
        for k in ('val_iou', 'val_iou_prev', 'val_iou_gain', 'val_pres_acc', 'val_attn_mass', 'train_loss', 'train_loss_attn', 'val_loss',
              'val_assoc_pred', 'val_assoc_prev', 'val_assoc_attn', 'val_assoc_gain', 'val_assoc_head', 'val_assoc_app', 'val_assoc_head_gain',
              'train_loss_assoc', 'val_loss_assoc'):
            assert all(np.isfinite(r[k]) for r in log), k
        assert log[0]['train_n_tracks'] > 0 and 0 < log[0]['val_present_frac'] <= 1
        assert log[0]['val_assoc_n'] > 0 and log[0]['select'] == 'val_assoc_head'
        assert log[0]['train_loss_assoc'] > 0
        main(_args(root, out, epochs=3, resume=True), det)                            # reanuda: solo la época 2
        log = [json.loads(l) for l in open(os.path.join(out, 'log.txt'))]
        assert [r['epoch'] for r in log] == [0, 1, 2]
        ck = torch.load(os.path.join(out, 'best.pth'), map_location='cpu')
        m = TrackHeads(); m.load_state_dict(ck['model'])


def test_resume_guards():
    """No se reanuda con otro --select (los mejores valores no son comparables); tras una parada temprana, relanzar no reentrena."""
    with tempfile.TemporaryDirectory() as d:
        root, out = os.path.join(d, 'data'), os.path.join(d, 'out')
        make_mot_root(root)
        det = StubDetector()
        main(_args(root, out, epochs=1, select='iou'), det)
        try:
            main(_args(root, out, epochs=2, resume=True), det)
            raise AssertionError('debía negarse a reanudar con otro --select')
        except SystemExit as e:
            assert '--select' in str(e)
        out2 = os.path.join(d, 'out2')
        main(_args(root, out2, epochs=5, patience=1, min_delta=10.0), det)           # min_delta enorme: ninguna época mejora, para tras la 1.ª
        n = len(open(os.path.join(out2, 'log.txt')).readlines())
        assert n == 1, n
        main(_args(root, out2, epochs=5, patience=1, min_delta=10.0, resume=True), det)
        assert len(open(os.path.join(out2, 'log.txt')).readlines()) == n


def test_diagnose_smoke():
    with tempfile.TemporaryDirectory() as d:
        root, out = os.path.join(d, 'data'), os.path.join(d, 'out')
        make_mot_root(root)
        det = StubDetector()
        main(_args(root, out, epochs=1), det)
        a = diag.parser().parse_args(['--ckpt', os.path.join(out, 'best.pth'), '--device', 'cpu', '--num-workers', '0', '--k-max', '8'])
        res = diag.main(a, det)
        t = res['total']
        assert t['n'] > 0 and 0 <= t['iou_prev'] <= 1 and 0 <= t['iou_pred'] <= 1
        assert t['iou_det_techo'] > 0.85 and t['det_encontrada'] == 1.0            # el detector simulado usa la caja GT con ruido pequeño
        assert sum(r.get('n', 0) for k, r in res.items() if k.startswith('k')) == t['n']
        assert res['cruces'].get('n', 0) + res['sin_cruce']['n'] == t['n'] and t['n_asoc'] > 0 and 0 <= t['asoc_pred'] <= 1
        assert res['presencia']['n'] >= t['n'] and os.path.isfile(os.path.join(out, 'diagnose.json'))
        assert 0 <= t['asoc_cabeza'] <= 1 and 0 <= t['asoc_apariencia'] <= 1 and 'w_iou' in res['pesos'] and 'recall' in res['ninguna']
        # checkpoint anterior al Associator (v1/v2): se diagnostica sin las columnas de la cabeza
        ck = torch.load(os.path.join(out, 'best.pth'), map_location='cpu')
        ck['args'].pop('w_assoc'); ck['model'] = {k: v for k, v in ck['model'].items() if not k.startswith('assoc.')}
        old = os.path.join(out, 'old.pth'); torch.save(ck, old)
        res = diag.main(diag.parser().parse_args(['--ckpt', old, '--device', 'cpu', '--num-workers', '0', '--out', os.path.join(out, 'd2.json')]), det)
        assert res['total']['asoc_cabeza'] is None and res['total']['asoc_pred'] is not None and 'pesos' not in res


if __name__ == '__main__':
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        fn()
        print('ok', name)
    print(f'{len(tests)} tests OK')
