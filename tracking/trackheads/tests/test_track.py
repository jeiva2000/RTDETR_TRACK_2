"""Tests del tracker en línea de TrackHeads (tracking/trackheads/track.py) en CPU: asignación Hungarian con "ninguna" y un humo sobre el
DanceTrack sintético de test_train con un detector simulado (cajas GT con ruido), incluida la evaluación con TrackEval.
Ejecución: python -m tracking.trackheads.tests.test_track   (o pytest tracking/trackheads/tests)"""
import os, sys, tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from tracking.trackheads.tests.test_train import StubDetector, _args, make_mot_root
from tracking.trackheads.track import assign, main, parser
from tracking.trackheads.train import main as train_main
from tracking.baseline.mot_io import read_mot, read_seqinfo
from tracking.video.frames import load_gt


def test_assign():
    score = torch.tensor([[0.9, 0.8], [0.85, 0.1], [0.1, 0.1]])
    # global: 0->1, 1->0 (0.8 + 0.85 > 0.9 + 0.1); el 2 se queda en "ninguna" (0.1 < 0.3)
    assert assign(score, torch.full((3,), 0.3)).tolist() == [1, 0, -1]
    assert assign(score, torch.full((3,), 0.95)).tolist() == [-1, -1, -1]
    assert assign(torch.zeros(0, 2), torch.zeros(0)).tolist() == [] and assign(torch.zeros(2, 0), torch.zeros(2)).tolist() == [-1, -1]


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
        for m in ('head', 'iou'):
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
