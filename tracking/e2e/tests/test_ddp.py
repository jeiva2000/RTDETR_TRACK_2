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


def _stub_fwd(x):
    """Detector falso determinista: tres cajas fijas con score 0.88 en cada imagen."""
    B = x.shape[0]
    boxes = torch.tensor([[0.2, 0.5, 0.15, 0.55], [0.45, 0.5, 0.15, 0.55], [0.7, 0.5, 0.15, 0.55]]).expand(B, 3, 4)
    return torch.full((B, 3, 1), 2.0), boxes


def _deta_worker(rank, world, tmp, seqs, port, q):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank))
    from tracking.e2e import dist_utils as du
    from tracking.joint.evaluate import eval_deta
    du.init_distributed()
    res = eval_deta(_stub_fwd, tmp + '/val', seqs, 2, 0.5, tmp + '/work_dist', num_workers=0, device='cpu', extra_thrs=[0.7], dist=du)
    q.put((rank, res)); du.cleanup()


def test_eval_deta_sharded_matches_single_process():
    """Regresión del bloqueo de H1 en el clúster: el DetA se calculaba con TODAS las secuencias en CADA proceso y todos escribían a la vez en la misma
    carpeta; los resultados podían diferir entre procesos, la decisión de parar el entrenamiento también, y DDP se quedaba esperando (hasta el timeout).
    Ahora cada proceso evalúa sus secuencias y rank 0 puntúa y difunde: ambos procesos deben devolver EL MISMO resultado, igual al de un solo proceso."""
    import random
    from tracking.joint.evaluate import eval_deta
    with tempfile.TemporaryDirectory() as tmp:
        seqs = []
        for i, n in enumerate([3, 2, 3, 1]):                      # secuencias distintas: el reparto entre procesos no es simétrico
            make_sequence(tmp + '/val', f'seq{i}', n=12, persons=n); seqs.append(f'seq{i}')
        ref = eval_deta(_stub_fwd, tmp + '/val', seqs, 2, 0.5, tmp + '/work_single', num_workers=0, device='cpu', extra_thrs=[0.7])
        ctx = torch.multiprocessing.get_context('spawn')
        q = ctx.Queue(); port = random.randint(20000, 40000)
        procs = [ctx.Process(target=_deta_worker, args=(r, 2, tmp, seqs, port, q)) for r in range(2)]
        [p.start() for p in procs]
        got = dict(q.get(timeout=300) for _ in range(2))
        [p.join(timeout=60) for p in procs]
        assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
        assert got[0] == got[1], f'los procesos deben devolver lo mismo: {got}'
        assert got[0] == ref, f'repartido != un proceso: {got[0]} vs {ref}'
        assert ref['n_det'] > 0 and 0 < ref['DetA'] < 100, ref


def test_ddp_two_processes_cpu(extra_ov=()):
    from src.core import YAMLConfig
    from tracking.video.extractor import DEFAULT_CONFIG
    with tempfile.TemporaryDirectory() as tmp:
        make_sequence(tmp + '/train', 'seqA'); make_sequence(tmp + '/val', 'seqB')
        torch.manual_seed(0)
        torch.save({'model': YAMLConfig(DEFAULT_CONFIG, PResNet=dict(pretrained=False)).model.state_dict()}, tmp + '/det.pth')
        cmd = [sys.executable, '-W', 'ignore', '-m', 'torch.distributed.run', '--standalone', '--local-addr', '127.0.0.1', '--nproc_per_node=2', '-m', 'tracking.e2e.train_joint',
               '--det-ckpt', tmp + '/det.pth', '--train-roots', tmp + '/train', '--val-root', tmp + '/val', '--out', tmp + '/run',
               '--epochs', '2', '--samples-per-epoch', '4', '--clips-per-gpu', '1', '--val-clips', '4', '--amp', 'none', '--num-workers', '0',
               '--warmup-steps', '1', '--print-freq', '1', '--deta-tol', '100', '--deta-thr', '0.0', '--deta-extra-thrs', '--deta-stride', '2',
               '--det-freeze-steps', '1', '--min-iou', '0', '--check-sync', '-u', 'train.T=3', 'train.max_stride=1', 'train.p_skip=0', *extra_ov]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, f'DDP falló:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}'
        assert '2 GPU(s), 1 clip(s) por GPU, lote efectivo 2 clips' in r.stdout, 'debe informar de 2 procesos y lote efectivo 2'
        assert r.stdout.count('sincronización de réplicas') == 2, 'comprobación de réplicas en cada época'
        for line in r.stdout.splitlines():
            if line.startswith('sincronización de réplicas'):
                assert float(line.split()[-1]) < 1e-5, line
        recs = [json.loads(l) for l in open(tmp + '/run/log.txt')]
        assert len(recs) == 2 and all(np.isfinite(x['train_det']) and np.isfinite(x['train_e2e']) for x in recs)
        motion, h3 = any('motion.mode=gru' in o for o in extra_ov), any('h3.enabled=true' in o for o in extra_ov)
        names = ('assign', 'app', 'motion') if h3 else ('assoc', 'birth', 'motion')        # partes de la pérdida e2e (H3: assign, app, motion)
        vnames = ('app_loss', 'motion_loss') if h3 else ('birth_loss', 'motion_nll')
        for x in recs:   # el e2e se registra también por partes (sin ponderar); la de movimiento solo existe en H2/H3
            assert all(np.isfinite(x[f'train_{k}']) and x[f'train_{k}'] >= 0 for k in names[:2]), x
            assert (f'train_{names[2]}' in x) == (motion or h3) and (f'val_{vnames[1]}' in x) == (motion or h3) and np.isfinite(x[f'val_{vnames[0]}']), x
        if h3:
            assert all('val_ids_ratio' in x and 'ss' in x for x in recs) and recs[-1]['ss'] == 1.0, 'el registro incluye la fragmentación de la validación y la proporción de clips cerrados'
        assert f'{names[0]} ' in r.stdout and f'{names[1]} ' in r.stdout and ((f'{names[2]} ' in r.stdout) == (motion or h3)), 'las líneas de paso deben mostrar las partes'
        assert os.path.isfile(tmp + '/run/e2e_last.pth') and os.path.isfile(tmp + '/run/detector_last.pth')


def test_ddp_two_processes_cpu_motion():
    """Igual con el H2 (GRU de movimiento): sus parámetros también deben mantenerse idénticos entre réplicas."""
    test_ddp_two_processes_cpu(('motion.mode=gru', 'motion.dim=32'))


def test_ddp_two_processes_cpu_h3():
    """Igual con el H3 (asociación relacional): sus parámetros (atención, gates, Sinkhorn) también deben mantenerse idénticos entre réplicas."""
    test_ddp_two_processes_cpu(('h3.enabled=true', 'h3.dim=64', 'h3.heads=4', 'h3.ffn=128', 'h3.gru_dim=32', 'h3.app_dim=32', 'h3.K_hist=4', 'h3.det_thr=0.0',
                                'state.max_tracks=8', 'tokens.top_k=20', 'train.ss_max=1.0', 'train.ss_warmup=1'))      # la 2.ª época corre con todos los clips en bucle cerrado


if __name__ == '__main__':
    test_eval_deta_sharded_matches_single_process(); print('ok   test_eval_deta_sharded_matches_single_process')
    test_ddp_two_processes_cpu(); print('ok   test_ddp_two_processes_cpu')
    test_ddp_two_processes_cpu_motion(); print('ok   test_ddp_two_processes_cpu_motion')
    test_ddp_two_processes_cpu_h3(); print('ok   test_ddp_two_processes_cpu_h3')
