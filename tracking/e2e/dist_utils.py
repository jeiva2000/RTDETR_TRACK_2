"""Utilidades de DDP para el entrenamiento e2e (torchrun). Todo funciona también sin entorno distribuido (world = 1).
Lanzamiento: torchrun --standalone --nproc_per_node=N -m tracking.e2e.train_joint ...  (cada proceso usa una GPU; en CPU usa gloo)."""
import datetime, os
import torch
import torch.distributed as dist


def init_distributed(timeout_hours=3.0):
    """Inicializa el grupo si torchrun definió WORLD_SIZE > 1. Devuelve (rank, world, local_rank, device)."""
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank, local = int(os.environ.get('RANK', 0)), int(os.environ.get('LOCAL_RANK', 0))
    if world > 1 and not dist.is_initialized():
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend, timeout=datetime.timedelta(hours=timeout_hours))   # la evaluación de rank 0 puede ser larga
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = f'cuda:{local}'
    else:
        device = 'cpu'
    return rank, world, local, device


def active(): return dist.is_available() and dist.is_initialized()
def rank(): return dist.get_rank() if active() else 0
def world(): return dist.get_world_size() if active() else 1
def is_main(): return rank() == 0
def barrier():
    if active():
        dist.barrier()


def all_gather_object(obj):
    """Lista con el objeto de cada proceso (en orden de rank)."""
    if not active():
        return [obj]
    out = [None] * world()
    dist.all_gather_object(out, obj)
    return out


def sum_dicts(d):
    """Suma elementwise los diccionarios numéricos (escalares) de todos los procesos."""
    return {k: sum(x[k] for x in all_gather_object(d)) for k in d}


def broadcast_object(obj, src=0):
    if not active():
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=src)
    return box[0]


def params_checksum(module):
    """Suma de los parámetros (float64) para comprobar que las réplicas siguen sincronizadas."""
    return float(sum(p.detach().double().sum() for p in module.parameters()))


def cleanup():
    if active():
        dist.destroy_process_group()
