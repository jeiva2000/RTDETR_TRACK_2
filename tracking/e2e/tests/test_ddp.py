"""Test de DDP del entrenamiento descongelado: 2 procesos en CPU (gloo) sobre un mini dataset generado aquí mismo y un detector con pesos aleatorios.
Comprueba que arranca, que cada proceso procesa clips y que las réplicas se mantienen idénticas tras el paso del optimizador (--check-sync), y que solo
el proceso principal escribe el log y los checkpoints. Ejecución: python -m tracking.e2e.tests.test_ddp   (~1-2 min en CPU)"""
import json, os, subprocess, sys, tempfile
import numpy as np
import torch
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'rtdetrv2_pytorch')); sys.path.insert(0, ROOT)


def make_sequence(root, name, n=30, persons=3):
    d = os.path.join(root, name); os.makedirs(d + '/img1'); os.makedirs(d + '/gt')
    open(d + '/seqinfo.ini', 'w').write(f'[Sequence]\nname={name}\nimDir=img1\nframeRate=20\nseqLength={n}\nimWidth=160\nimHeight=90\nimExt=.jpg\n')
    rows = []
    for f in range(1, n + 1):
        im = np.full((90, 160, 3), 90, np.uint8)
        for i in range(persons):
            x, y = 10 + 40 * i + f, 20
            im[y:y + 50, x:x + 20] = (60 * i + 40, 200 - 50 * i, 100)
            rows.append([f, i, x, y, 20, 50, 1, 1, 1])
        Image.fromarray(im).save(f'{d}/img1/{f:08d}.jpg')
    np.savetxt(d + '/gt/gt.txt', np.array(rows), delimiter=',', fmt='%d')


def test_ddp_two_processes_cpu():
    from src.core import YAMLConfig
    from tracking.video.extractor import DEFAULT_CONFIG
    with tempfile.TemporaryDirectory() as tmp:
        make_sequence(tmp + '/train', 'seqA'); make_sequence(tmp + '/val', 'seqB')
        torch.manual_seed(0)
        torch.save({'model': YAMLConfig(DEFAULT_CONFIG, PResNet=dict(pretrained=False)).model.state_dict()}, tmp + '/det.pth')
        cmd = [sys.executable, '-W', 'ignore', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', '-m', 'tracking.e2e.train_joint',
               '--det-ckpt', tmp + '/det.pth', '--train-roots', tmp + '/train', '--val-root', tmp + '/val', '--out', tmp + '/run',
               '--epochs', '2', '--samples-per-epoch', '4', '--clips-per-gpu', '1', '--val-clips', '4', '--amp', 'none', '--num-workers', '0',
               '--warmup-steps', '1', '--print-freq', '1', '--deta-tol', '100', '--deta-thr', '0.0', '--deta-extra-thrs', '--deta-stride', '2',
               '--det-freeze-steps', '1', '--min-iou', '0', '--check-sync', '-u', 'train.T=3', 'train.max_stride=1', 'train.p_skip=0']
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, f'DDP falló:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}'
        assert '2 GPU(s), 1 clip(s) por GPU, lote efectivo 2 clips' in r.stdout, 'debe informar de 2 procesos y lote efectivo 2'
        assert r.stdout.count('sincronización de réplicas') == 2, 'comprobación de réplicas en cada época'
        for line in r.stdout.splitlines():
            if line.startswith('sincronización de réplicas'):
                assert float(line.split()[-1]) < 1e-5, line
        recs = [json.loads(l) for l in open(tmp + '/run/log.txt')]
        assert len(recs) == 2 and all(np.isfinite(x['train_det']) and np.isfinite(x['train_e2e']) for x in recs)
        assert os.path.isfile(tmp + '/run/e2e_last.pth') and os.path.isfile(tmp + '/run/detector_last.pth')


if __name__ == '__main__':
    test_ddp_two_processes_cpu(); print('ok   test_ddp_two_processes_cpu')
