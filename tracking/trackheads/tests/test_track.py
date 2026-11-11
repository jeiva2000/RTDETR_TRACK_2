"""Tests del tracker en línea de TrackHeads (tracking/trackheads/track.py) en CPU: asignación Hungarian con "ninguna" y un humo sobre el
DanceTrack sintético de test_train con un detector simulado (cajas GT con ruido), incluida la evaluación con TrackEval.
Ejecución: python -m tracking.trackheads.tests.test_track   (o pytest tracking/trackheads/tests)"""
import os, sys, tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from tracking.trackheads.tests.test_train import StubDetector, _args, make_mot_root
from tracking.trackheads.track import Tracker, assign, main, parse_config, parser
from tracking.trackheads.train import main as train_main
from tracking.baseline.mot_io import read_mot, read_seqinfo
from tracking.video.frames import load_gt


def test_assign():
    score = torch.tensor([[0.9, 0.8], [0.85, 0.1], [0.1, 0.1]])
    # global: 0->1, 1->0 (0.8 + 0.85 > 0.9 + 0.1); el 2 se queda en "ninguna" (0.1 < 0.3)
    assert assign(score, torch.full((3,), 0.3)).tolist() == [1, 0, -1]
    assert assign(score, torch.full((3,), 0.95)).tolist() == [-1, -1, -1]
    assert assign(torch.zeros(0, 2), torch.zeros(0)).tolist() == [] and assign(torch.zeros(2, 0), torch.zeros(2)).tolist() == [-1, -1]


def _tracker(**kw):
    a = parser().parse_args(['--ckpt', 'x', '--out', 'x'])
    spec = 'c=iou' + ''.join(f',{k}={v}' for k, v in kw.items())
    name, cfg = parse_config(spec, a)
    assert name == 'c' and cfg.mode == 'iou'
    return Tracker(cfg, model=None, dim=4)


def _run(tr, boxes_per_frame, scores=None):
    """boxes_per_frame: lista de listas de cajas cxcywh -> lista (por frame) de ids escritos (en el orden de las candidatas)."""
    res = []
    for f, bs in enumerate(boxes_per_frame, 1):
        box = torch.tensor(bs, dtype=torch.float32).reshape(-1, 4)
        sc = torch.tensor(scores[f - 1] if scores else [0.9] * len(bs), dtype=torch.float32)
        out = dict(tr.step(f, torch.zeros(len(bs), 4), box, sc, None))
        res.append(sorted(out))
    return res


def test_min_hits_and_out_thresh():
    p = [0.3, 0.5, 0.1, 0.2]
    # sin confirmación: se escribe desde el primer frame
    assert _run(_tracker(), [[p]] * 4) == [[1]] * 4
    # min_hits=3: los frames 1-3 son "early" (como OC-SORT) y se escriben; un track que nace en el frame 5 espera 3 asignaciones seguidas
    q = [0.8, 0.5, 0.1, 0.2]
    w = _run(_tracker(min_hits=3), [[p]] * 4 + [[p, q]] * 4)
    assert w[:4] == [[1]] * 4 and w[4] == [1] and w[5] == [1] and w[6] == [1, 2] and w[7] == [1, 2]
    # un track sin confirmar muere al fallar un frame: el que nace en 5, falta en 6 y vuelve en 7 es otro id (3), sin confirmar aún
    w = _run(_tracker(min_hits=3), [[p]] * 4 + [[p, q], [p], [p, q], [p, q], [p, q]])
    assert w[6] == [1] and w[8] == [1, 3]
    # out_thresh: la caja de score bajo no se escribe, pero el track sigue vivo con el mismo id
    w = _run(_tracker(out_thresh=0.5), [[p]] * 3, scores=[[0.9], [0.4], [0.9]])
    assert w == [[1], [], [1]]


def test_motion():
    # una persona que avanza 0.03 por frame en x con cajas de ancho 0.1 (IoU 0.54 entre frames seguidos): tras 3 frames sin verla, el IoU
    # con la última caja es 0 y nace otro id; con velocidad constante, la caja extrapolada la recupera
    seq = [[[0.1 + 0.03 * f, 0.5, 0.1, 0.2]] for f in range(5)] + [[]] * 3 + [[[0.1 + 0.03 * f, 0.5, 0.1, 0.2]] for f in range(8, 11)]
    still = _run(_tracker(), seq)
    moving = _run(_tracker(motion=1), seq)
    assert still[-1] != [1] and moving[-1] == [1], (still, moving)
    tr = _tracker(motion=1, motion_max=2)
    _run(tr, seq[:5])
    assert abs(tr.vel[0, 0].item() - 0.03) < 1e-5 and abs(tr.ref_box(10)[0, 0].item() - (0.22 + 0.06)) < 1e-5    # extrapola como mucho 2 frames


def test_emb_update():
    # dos personas: A fija en x=0.3; B se acerca y solapa con A en los frames 3-4. emb de A = +1, de B = -1; en el cruce, la detección
    # de A trae un emb mezclado (0).
    def frames(tr):
        far, near = [0.8, 0.5, 0.1, 0.2], [0.33, 0.5, 0.1, 0.2]
        A = [0.3, 0.5, 0.1, 0.2]
        for f, (b, ea) in enumerate([(far, 1.), (far, 1.), (near, 0.), (near, 0.)], 1):
            box = torch.tensor([A, b]); emb = torch.tensor([[ea] * 4, [-1.] * 4])
            tr.step(f, emb, box, torch.tensor([0.9, 0.9]), None)
        return tr.emb[tr.ids.index(1), 0].item()
    assert frames(_tracker()) == 0.0                                                   # last: se contamina
    assert frames(_tracker(emb_update='clean')) == 1.0                                 # clean: conserva la apariencia previa al cruce
    assert abs(frames(_tracker(emb_update='ema', emb_ema=0.5)) - 0.25) < 1e-6          # 1 -> 1 -> 0.5 -> 0.25
    assert frames(_tracker(emb_update='clean_ema', emb_ema=0.5)) == 1.0
    try:
        _tracker(emb_update='otro')
        rejected = False
    except AssertionError as e:
        rejected = 'emb_update desconocido' in str(e)
    assert rejected, 'debía rechazar un emb_update desconocido'


class SeqStub(StubDetector):
    """Inferencia sin GT para una secuencia: devuelve las cajas GT (con ruido) del frame siguiente en cada imagen recibida."""
    def __init__(self, gt):
        super().__init__(K=8)
        self.gt, self.f = gt, 0

    def infer(self, images):
        gts = []
        for _ in range(images.shape[0]):
            self.f += 1
            gts.append(self.gt.get(self.f, torch.zeros(0, 5)))
        q, _, feats = self.detect(images, gts)
        return q, feats


def test_track_smoke_and_eval():
    with tempfile.TemporaryDirectory() as d:
        root, out = os.path.join(d, 'data'), os.path.join(d, 'out')
        make_mot_root(root)
        train_main(_args(root, os.path.join(d, 'th'), epochs=1), StubDetector())
        sd = os.path.join(root, 'val', 'va0')
        info = read_seqinfo(sd)
        a = parser().parse_args(['--ckpt', os.path.join(d, 'th', 'best.pth'), '--out', out, '--device', 'cpu', '--num-workers', '0',
                                 '--batch', '4', '--max-lost', '10'])
        seqs = main(a, SeqStub(load_gt(sd, info)))
        assert seqs == ['va0']
        assert sorted(os.listdir(out)) == sorted(c.split('=')[0] for c in a.configs)
        for m in os.listdir(out):
            r = read_mot(os.path.join(out, m, 'va0.txt'))
            assert len(r) > 0 and r[:, 0].min() >= 1 and r[:, 0].max() <= info['length']
            for f in set(r[:, 0].tolist()):                                           # un id aparece una vez por frame como mucho
                ids = r[r[:, 0] == f, 1]
                assert len(ids) == len(set(ids.tolist()))
        # el tracker por IoU sobre cajas GT casi exactas: 3 personas -> como mucho una identidad nueva por la persona que sale y vuelve
        assert len(set(read_mot(os.path.join(out, 'iou', 'va0.txt'))[:, 1].tolist())) <= 4
        from tracking.baseline.evaluate import trackeval_run
        res = trackeval_run(os.path.join(root, 'val'), os.path.join(out, 'iou'), {'va0': info['length']})
        assert res['HOTA'] > 70, res


if __name__ == '__main__':
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        fn()
        print('ok', name)
    print(f'{len(tests)} tests OK')
