"""Tests del ajuste conjunto de TrackHeads con el RT-DETRv2 descongelado (tracking/trackheads/train_joint.py), en CPU con el detector real del
repo y pesos aleatorios sobre el DanceTrack sintético de test_train: el gradiente de la pérdida conjunta llega al backbone, al encoder y al
decoder; un humo de main() (DetA inicial, 1 época, checkpoints del detector y de TrackHeads) y que diagnose.py carga lo que deja.
Ejecución: python -m tracking.trackheads.tests.test_train_joint   (~2-4 min)"""
import json, os, sys, tempfile

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'rtdetrv2_pytorch')); sys.path.insert(0, ROOT)
from src.core import YAMLConfig
from tracking.joint.train import Detector
from tracking.trackheads import diagnose as diag
from tracking.trackheads.data import PairDataset, collate
from tracking.trackheads.tests.test_train import make_mot_root
from tracking.trackheads.train import build_model
from tracking.trackheads.train_joint import JointDetector, joint_loss, main, parser
from tracking.video.extractor import DEFAULT_CONFIG


def _det_ckpt(tmp):
    path = os.path.join(tmp, 'det.pth')
    if not os.path.exists(path):
        torch.manual_seed(0)
        torch.save({'model': YAMLConfig(DEFAULT_CONFIG, PResNet=dict(pretrained=False)).model.state_dict()}, path)
    return path


def _args(root, out, det, **kw):
    a = parser().parse_args(['--root', root, '--det-ckpt', det, '--out', out, '--device', 'cpu', '--amp', 'none', '--num-workers', '0',
                             '--batch', '1', '--samples', '2', '--val-samples', '2', '--epochs', '1', '--warmup', '1', '--print-every', '1',
                             '--k-max', '4', '--deta-stride', '10', '--deta-extra-thrs', '--det-freeze-steps', '0', '--lr-det', '1e-3', '--lr', '1e-3',
                             '--min-iou', '0', '--det-thr', '0'])          # con pesos aleatorios ninguna query supera el IoU ni el score: sin filtros hay tracks
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_gradient_reaches_whole_detector():
    with tempfile.TemporaryDirectory() as d:
        root = os.path.join(d, 'data'); make_mot_root(root)
        a = _args(root, os.path.join(d, 'out'), _det_ckpt(d))
        det = Detector(YAMLConfig(DEFAULT_CONFIG, PResNet=dict(pretrained=False)), a.det_ckpt, 'cpu', 'none', train_encoder=True, train_backbone=True)
        model = build_model(a)
        batch = collate([PairDataset(root, 'train', 1, 4, 1, seed=0)[0]])
        loss, p = joint_loss(det, model, batch, a, 'cpu', 0, 0)
        assert torch.isfinite(loss) and p['det'] > 0 and p['th'] > 0 and set(p) >= {'assoc', 'l1', 'pres', 'attn'}
        loss.backward()
        for part in ('backbone', 'encoder', 'decoder'):
            assert any(pp.grad is not None and pp.grad.abs().sum() > 0 for pp in det.params(part)), part
        assert all(pp.grad is None for pp in det.model.backbone.conv1.parameters()), 'la primera capa del backbone sigue congelada'
        assert any(pp.grad is not None and pp.grad.abs().sum() > 0 for pp in model.parameters())
        # la evaluación con la interfaz de FrozenDetector no deja grafo y devuelve lo mismo que detect()
        q, tid, feats = JointDetector(det, a.top_k).detect(batch['img_t'], batch['gt_t'])
        assert q['emb'].shape == (1, a.top_k, 256) and tid.shape == (1, a.top_k) and len(feats) == 3 and not feats[0].requires_grad


def test_main_smoke_and_diagnose():
    with tempfile.TemporaryDirectory() as d:
        root, out = os.path.join(d, 'data'), os.path.join(d, 'out')
        make_mot_root(root)
        a = _args(root, out, _det_ckpt(d), deta_stop=False)                       # con pesos aleatorios el DetA es 0: sin parada por DetA
        before = torch.load(a.det_ckpt, map_location='cpu')['model']
        main(a)
        for f in ('best.pth', 'last.pth', 'detector_best.pth', 'detector_last.pth', 'log.txt', 'args.json'):
            assert os.path.isfile(os.path.join(out, f)), f
        log = [json.loads(l) for l in open(os.path.join(out, 'log.txt'))]
        assert len(log) == 1 and 'DetA' in log[0] and 'val_assoc_head' in log[0] and log[0]['select'] == 'val_assoc_head' and log[0]['improved']
        after = torch.load(os.path.join(out, 'detector_last.pth'), map_location='cpu')['model']
        changed = lambda pref: any(not torch.equal(before[k], after[k]) for k in before if k.startswith(pref) and before[k].is_floating_point())
        assert changed('backbone.res_layers') and changed('encoder.') and changed('decoder.'), 'el detector se entrena'
        assert not changed('backbone.conv1')
        ck = torch.load(os.path.join(out, 'best.pth'), map_location='cpu')
        assert ck['args']['det_ckpt'] == os.path.join(out, 'detector_best.pth') and 'w_assoc' in ck['args']
        # diagnose.py carga best.pth y su detector solo (FrozenDetector con detector_best.pth)
        res = diag.main(diag.parser().parse_args(['--ckpt', os.path.join(out, 'best.pth'), '--device', 'cpu', '--num-workers', '0', '--no-amp', '--val-samples', '1']))
        assert 'pesos' in res and res['presencia']['n'] >= 0
        # relanzar con --resume tras terminar las épocas: no reentrena (el log no crece)
        main(_args(root, out, a.det_ckpt, deta_stop=False, resume=True))
        assert len(open(os.path.join(out, 'log.txt')).readlines()) == 1


if __name__ == '__main__':
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        fn()
        print('ok', name)
    print(f'{len(tests)} tests OK')
