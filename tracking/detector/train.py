"""Igual que rtdetrv2_pytorch/tools/train.py, pero registra DetSolverES (early stopping) como solver de la tarea 'detection'.
Uso: igual que tools/train.py (se lanza desde rtdetrv2_pytorch/ o con torchrun)."""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'rtdetrv2_pytorch'))
sys.path.insert(0, ROOT)

import argparse
from src.misc import dist_utils
from src.core import YAMLConfig, yaml_utils
from src.solver import TASKS
from tracking.detector.solver_es import DetSolverES

TASKS['detection'] = DetSolverES


def main(args):
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    assert not all([args.tuning, args.resume]), 'Only support from_scrach or resume or tuning at one time'
    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() if k not in ['update', ] and v is not None})
    cfg = YAMLConfig(args.config, **update_dict)
    print('cfg: ', cfg.__dict__)
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.val() if args.test_only else solver.fit()
    dist_utils.cleanup()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', type=str, required=True)
    p.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    p.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    p.add_argument('-d', '--device', type=str, help='device')
    p.add_argument('--seed', type=int, help='exp reproducibility')
    p.add_argument('--use-amp', action='store_true', help='auto mixed precision training')
    p.add_argument('--output-dir', type=str, help='output directoy')
    p.add_argument('--summary-dir', type=str, help='tensorboard summry')
    p.add_argument('--test-only', action='store_true', default=False)
    p.add_argument('-u', '--update', nargs='+', help='update yaml config')
    p.add_argument('--print-method', type=str, default='builtin')
    p.add_argument('--print-rank', type=int, default=0)
    p.add_argument('--local-rank', type=int)
    main(p.parse_args())
