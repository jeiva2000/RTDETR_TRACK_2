"""Tests del descongelado del detector en train_joint: qué parámetros del RT-DETR son entrenables según --train-backbone / --train-encoder (la primera capa del backbone y sus
normalizaciones siguen congeladas, como en el entrenamiento por defecto de RT-DETRv2), que el gradiente llega al backbone incluso con el encoder congelado, y un entrenamiento
real de 1 proceso que cambia (o no) los pesos del backbone y del encoder según las banderas. CPU, detector con pesos aleatorios.
Ejecución: python -m tracking.e2e.tests.test_detector_unfreeze   (~2-3 min)"""
import json, os, subprocess, sys, tempfile
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'rtdetrv2_pytorch')); sys.path.insert(0, ROOT)
from src.core import YAMLConfig
from tracking.e2e.tests.test_ddp import make_sequence
from tracking.joint.train import Detector
from tracking.video.extractor import DEFAULT_CONFIG


def _detector(tmp, **kw):
    torch.manual_seed(0)
    cfg = YAMLConfig(DEFAULT_CONFIG, PResNet=dict(pretrained=False))
    if not os.path.exists(tmp + '/det.pth'):
        torch.save({'model': cfg.model.state_dict()}, tmp + '/det.pth')
    return Detector(YAMLConfig(DEFAULT_CONFIG, PResNet=dict(pretrained=False)), tmp + '/det.pth', 'cpu', 'none', **kw)


def test_trainable_parameters():
    with tempfile.TemporaryDirectory() as tmp:
        d = _detector(tmp)                                                          # por defecto del Detector: congelados (flujo antiguo de tracking/joint)
        assert not d.params('backbone') and not d.params('encoder') and d.params('decoder')
        d = _detector(tmp, train_encoder=True, train_backbone=True)
        names = {id(p): n for n, p in d.model.named_parameters()}
        bb = [names[id(p)] for p in d.params('backbone')]
        assert bb and not any(n.startswith('backbone.conv1') for n in bb), 'la primera capa del backbone sigue congelada (freeze_at=0)'
        assert any('res_layers.0' in n for n in bb) and any('res_layers.3' in n for n in bb), 'las 4 etapas residuales son entrenables'
        assert all(p.requires_grad for p in d.model.encoder.parameters()) and len(d.params('encoder')) == len(list(d.model.encoder.parameters()))
        assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in d.model.backbone.modules()), 'las normalizaciones del backbone son FrozenBatchNorm (sin parámetros)'
        d = _detector(tmp, train_encoder=True)
        assert d.params('encoder') and not d.params('backbone')
        d = _detector(tmp, train_backbone=True)
        assert d.params('backbone') and not d.params('encoder')


def test_gradient_reaches_backbone_and_eval_has_none():
    with tempfile.TemporaryDirectory() as tmp:
        x = torch.rand(1, 3, 640, 640)
        d = _detector(tmp, train_backbone=True)                                      # backbone entrenable con el encoder CONGELADO: el gradiente debe atravesar el encoder
        feats = d.features(x)
        sum(f.sum() for f in feats).backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in d.params('backbone'))
        assert all(p.grad is None for p in d.model.encoder.parameters())
        d = _detector(tmp, train_encoder=True, train_backbone=True)
        sum(f.sum() for f in d.features(x)).backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in d.params('encoder')) and all(p.grad is None for p in d.model.backbone.conv1.parameters())
        with torch.no_grad():                                                        # la evaluación va en no_grad: sin grafo aunque estén entrenables
            assert not any(f.requires_grad for f in d.features(x))
        d = _detector(tmp)
        assert not any(f.requires_grad for f in d.features(x)), 'congelados: sin grafo'


def _run(tmp, extra):
    out = f'{tmp}/run_' + '_'.join(a.strip('-') for a in extra) if extra else f'{tmp}/run_default'
    cmd = [sys.executable, '-W', 'ignore', '-m', 'tracking.e2e.train_joint', '--det-ckpt', tmp + '/det.pth', '--train-roots', tmp + '/train', '--val-root', tmp + '/val',
           '--out', out, '--epochs', '1', '--samples-per-epoch', '4', '--clips-per-gpu', '1', '--val-clips', '2', '--amp', 'none', '--num-workers', '0',
           '--warmup-steps', '1', '--print-freq', '1', '--deta-tol', '100', '--deta-thr', '0.0', '--deta-extra-thrs', '--deta-stride', '2', '--det-freeze-steps', '1',
           '--min-iou', '0', *extra, '-u', 'train.T=3', 'train.max_stride=1', 'train.p_skip=0']
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f'falló:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}'
    return out, r.stdout


def _changed(tmp, out):
    a, b = torch.load(tmp + '/det.pth')['model'], torch.load(out + '/detector_last.pth')['model']
    ch = lambda pre: [k for k in a if k.startswith(pre) and a[k].dtype.is_floating_point and not torch.equal(a[k], b[k])]
    return ch('backbone.'), ch('encoder.'), ch('decoder.')


def test_train_joint_unfreezes_by_default_and_flags_freeze():
    with tempfile.TemporaryDirectory() as tmp:
        make_sequence(tmp + '/train', 'seqA'); make_sequence(tmp + '/val', 'seqB')
        _detector(tmp)
        out, stdout = _run(tmp, [])                                                  # por defecto: backbone y encoder entrenables
        bb, enc, dec = _changed(tmp, out)
        assert 'detector entrenable: decoder' in stdout and 'encoder' in stdout.split('detector entrenable:')[1].splitlines()[0] and 'backbone' in stdout.split('detector entrenable:')[1].splitlines()[0]
        assert bb and enc and dec, 'por defecto cambian los pesos del backbone, del encoder y del decoder'
        assert not any(k.startswith('backbone.conv1') for k in bb), 'la primera capa del backbone no cambia'
        out, stdout = _run(tmp, ['--no-train-backbone', '--no-train-encoder'])
        bb, enc, dec = _changed(tmp, out)
        assert not bb and not enc and dec, 'congelados: solo cambia el decoder'
        assert 'backbone' not in stdout.split('detector entrenable:')[1].splitlines()[0]
        out, _ = _run(tmp, ['--no-train-backbone'])
        bb, enc, dec = _changed(tmp, out)
        assert not bb and enc and dec, 'solo encoder + decoder'
        rec = json.loads(open(out + '/log.txt').readline())
        assert rec['DetA'] >= 0


def test_deta_stop_is_optional():
    """Con una tolerancia imposible (el DetA nunca la cumple) el entrenamiento por defecto se detiene tras la 1.ª época; con --no-deta-stop sigue todas las épocas y guarda un mejor checkpoint."""
    with tempfile.TemporaryDirectory() as tmp:
        make_sequence(tmp + '/train', 'seqA'); make_sequence(tmp + '/val', 'seqB')
        _detector(tmp)
        base = ['--epochs', '2', '--deta-tol', '-1000', '--no-train-backbone', '--no-train-encoder']
        out, stdout = _run(tmp, base)
        assert len(open(out + '/log.txt').read().splitlines()) == 1 and 'PARADA' in stdout, 'por defecto: se detiene tras la 1.ª época'
        assert not os.path.isfile(out + '/e2e_best.pth'), 'ninguna época válida: no hay mejor checkpoint'
        out, stdout = _run(tmp, base + ['--no-deta-stop'])
        recs = [json.loads(l) for l in open(out + '/log.txt')]
        assert len(recs) == 2 and 'PARADA' not in stdout and 'AVISO: DetA' in stdout and all(not r['deta_ok'] for r in recs), 'sin parada: se entrenan las dos épocas y se avisa del DetA'
        assert os.path.isfile(out + '/e2e_best.pth') and os.path.isfile(out + '/detector_best.pth'), 'sin parada, el mejor checkpoint se elige entre todas las épocas'


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for t in tests:
        t(); print('ok  ', t.__name__)
    print(f'{len(tests)} tests correctos')
