"""Módulo e2e con el detector DESCONGELADO (ajuste conjunto): se entrenan a la vez el detector (por defecto decoder + encoder + backbone) y el módulo e2e, con extracción
en línea (sin caché). Como en el entrenamiento por defecto de RT-DETRv2, la primera capa del backbone y las normalizaciones del backbone siguen congeladas; todo el
detector va en eval (BatchNorm con estadísticas fijas: con lotes de frames de un mismo clip las del lote no son las de la población; ver tracking/joint/train.py, clase Detector).
--no-train-backbone / --no-train-encoder los dejan congelados (sin gradiente) para ablaciones o para ahorrar memoria y tiempo.
Pérdida = detección completa de RT-DETRv2 sobre los frames del clip + lambda_e2e * pérdida e2e con gradiente hacia el detector a través de los embeddings
(cajas y scores se desconectan). Grupos de parámetros: decoder a --lr-det (2e-5), encoder a --lr-encoder (por defecto = --lr-det), backbone a --lr-backbone (por defecto
--lr-det / 10, la misma proporción que RT-DETRv2: 1e-4 / 1e-5; sin validar) y módulo a train.lr de la config; un único recorte de gradiente (--clip-det) para el detector.
Control del detector: DetA en val cada época (por defecto la media de los umbrales 0.5/0.6/0.7, que es menos sensible a desplazamientos de calibración de
los scores que un solo umbral); si cae más de --deta-tol puntos por debajo del DetA inicial, se informa y se detiene. Con módulo desde cero, los primeros
--det-freeze-steps pasos solo aprende el módulo (todo el detector a lr 0).
DDP (varias GPUs): cada proceso (una GPU) procesa --clips-per-gpu clips distintos por paso (DistributedSampler) y los gradientes se sincronizan con DDP (el backbone y el
encoder entrenables se registran en JointModule: si no, DDP no los sincronizaría); el lote efectivo es clips-por-GPU x nº de GPUs y --samples-per-epoch cuenta clips de TODAS
las GPUs. La evaluación (DetA y precisión en vivo) se reparte entre procesos. Entrenar el backbone cuesta bastante más memoria y tiempo por paso (sin medir en GPU).
Lanzamiento: torchrun --standalone --nproc_per_node=N -m tracking.e2e.train_joint ...   (sin torchrun, una sola GPU).
Salidas en --out: detector_{best,last}.pth ({'model'}: modelo completo, backbone y encoder incluidos), e2e_{best,last}.pth (compatibles con tracking.e2e.infer), log.txt.
Uso (1 GPU): python -m tracking.e2e.train_joint --det-ckpt <best.pth> --out <dir> [--cfg tracking/configs/e2e.yml] [-u appearance.alpha_app=0.2] [--init-e2e <e2e best.pth>]"""
import argparse, json, os, random, sys, time
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'rtdetrv2_pytorch')); sys.path.insert(0, ROOT)
from src.core import YAMLConfig
from tracking.e2e import dist_utils as du
from tracking.e2e.config import load_config
from tracking.e2e.model import build_model
from tracking.e2e.train import has_motion, run_batch, val_names
from tracking.joint.evaluate import eval_deta
from tracking.joint.train import Detector
from tracking.video.clips import ImageClipDataset, collate_images
from tracking.video.extractor import DEFAULT_CONFIG, match_queries, topk_queries


class JointModule(torch.nn.Module):
    """Decoder del detector + módulo e2e en un solo nn.Module cuyo forward devuelve la pérdida total: así DDP sincroniza TODOS los gradientes con una única
    llamada de backward. Backbone y encoder (congelados, sin gradiente) viven en `det` y no se replican como parámetros entrenables."""

    def __init__(self, det, e2e, cfg, K, T, min_iou, lambda_e2e):
        super().__init__()
        self.dec, self.e2e = det.dec, e2e
        if det.train_backbone:       # registrados como submódulos para que DDP sincronice sus gradientes (det es un objeto normal, no un nn.Module)
            self.backbone = det.model.backbone
        if det.train_encoder:
            self.encoder = det.model.encoder
        self.det, self.cfg, self.K, self.T, self.min_iou, self.lam = det, cfg, K, T, min_iou, lambda_e2e

    def forward(self, images, gts, frames, ss, epoch, step):
        B, T, K = frames.shape[0], self.T, self.K
        dev = images.device
        targets = [dict(labels=torch.zeros(len(g), dtype=torch.long, device=dev), boxes=g[:, 1:5].to(dev).float()) for g in gts]
        out, emb = self.det.forward(images, targets)
        with torch.autocast(device_type=torch.device(dev).type, enabled=False):
            loss_dict = self.det.criterion(out, targets, epoch=epoch, step=step, global_step=step)
        loss_det = sum(loss_dict.values())
        q = topk_queries(out['pred_logits'].float(), out['pred_boxes'].float(), emb.float(), K)
        tid, _ = match_queries(self.det.criterion.matcher, q, gts, dev, self.min_iou)
        ebatch = dict(emb=q['emb'].view(B, T, K, -1), boxes=q['boxes'].detach().view(B, T, K, 4), scores=q['scores'].detach().view(B, T, K),
                      tid=tid.view(B, T, K).to(dev), frames=frames, cross=[None] * B)
        loss_e2e, st = run_batch(self.e2e, ebatch, self.cfg, dev, True, ss)
        return (loss_det + self.lam * loss_e2e, loss_det.detach(), loss_e2e.detach(), {k: float(v) for k, v in loss_dict.items()},
                st['parts'])      # partes de la pérdida e2e (H1/H2: asociación, nacimiento, movimiento; H3: assign, app, motion) sin ponderar, para el log


def control_value(d, metric):
    """DetA que controla la parada: a 0.6, o la media de los umbrales registrados."""
    return float(np.mean(list(d['DetA_by_thr'].values()))) if metric == 'mean' and 'DetA_by_thr' in d else float(d['DetA'])


def main(a):
    rank, world, local, dev = du.init_distributed()
    main_proc = rank == 0
    P = (lambda *x, **k: print(*x, **k)) if main_proc else (lambda *x, **k: None)       # solo el proceso principal escribe en el log
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)               # misma semilla para construir (DDP además difunde desde rank 0)
    if main_proc:
        os.makedirs(a.out, exist_ok=True)
    du.barrier()
    init = torch.load(a.init_e2e, map_location='cpu') if a.init_e2e else None
    cfg = load_config(a.cfg, a.update) if not init else load_config(None, init['cfg'])
    if init and a.update:
        from tracking.e2e.config import deep_update, parse_overrides
        deep_update(cfg, parse_overrides(a.update))
    assert cfg.e2e.enabled, 'e2e.enabled=false: el módulo está desactivado por configuración'
    assert cfg.train.burn_in == 0, 'el ajuste conjunto e2e del H1 no usa burn-in'
    T, K, B = cfg.train.T, cfg.tokens.top_k, a.clips_per_gpu
    det = Detector(YAMLConfig(a.config, PResNet=dict(pretrained=False)), a.det_ckpt, dev, a.amp, train_encoder=a.train_encoder, train_backbone=a.train_backbone)
    model = build_model(cfg, init['in_dim'] if init else 256).to(dev)
    if init:
        model.load_state_dict(init['model'])
    model.train()
    scratch = init is None
    P(f'módulo e2e: {"DESDE CERO" if scratch else "inicializado desde " + a.init_e2e}, {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parámetros | '
      f'{world} GPU(s), {a.clips_per_gpu} clip(s) por GPU, lote efectivo {a.clips_per_gpu * world} clips', flush=True)
    det_freeze = a.det_freeze_steps if a.det_freeze_steps is not None else (1000 if scratch else 0)
    ss_warmup = cfg.train.ss_warmup if scratch else 0
    matcher = det.criterion.matcher
    dec_params, enc_params, bb_params = det.params('decoder'), det.params('encoder'), det.params('backbone')
    det_params = dec_params + enc_params + bb_params
    lr_mult = {'none': 1.0, 'linear': float(world), 'sqrt': float(world) ** 0.5}[a.lr_scale]     # reescalado del lr con el lote efectivo (opcional, sin validar)
    if lr_mult != 1.0:
        P(f'lr reescalado x{lr_mult:.2f} (--lr-scale {a.lr_scale}, {world} GPUs)')
    a.lr_encoder = a.lr_det if a.lr_encoder is None else a.lr_encoder
    a.lr_backbone = a.lr_det / 10 if a.lr_backbone is None else a.lr_backbone
    P('detector entrenable: ' + ', '.join(f'{n} {sum(p.numel() for p in ps) / 1e6:.2f}M (lr {lr:.1e})' for n, ps, lr in
                                           (('decoder', dec_params, a.lr_det), ('encoder', enc_params, a.lr_encoder), ('backbone', bb_params, a.lr_backbone)) if ps), flush=True)

    def det_groups(params, lr):     # pesos con weight decay, sesgos y normalizaciones sin él (como RT-DETR); los grupos vacíos se omiten
        return [dict(params=ps, lr=lr * lr_mult, weight_decay=wd, base=lr * lr_mult, kind='det') for ps, wd in
                (([p for p in params if p.ndim > 1], 1e-4), ([p for p in params if p.ndim <= 1], 0.0)) if ps]

    groups = (det_groups(dec_params, a.lr_det) + det_groups(enc_params, a.lr_encoder) + det_groups(bb_params, a.lr_backbone) +
              [dict(params=list(model.parameters()), lr=cfg.train.lr * lr_mult, weight_decay=cfg.train.weight_decay, base=cfg.train.lr * lr_mult, kind='mod')])
    opt = torch.optim.AdamW(groups)
    scaler = torch.cuda.amp.GradScaler(enabled=(a.amp == 'fp16' and torch.device(dev).type == 'cuda'))
    roots = a.train_roots or [os.path.join(a.dancetrack, 'train1'), os.path.join(a.dancetrack, 'train2')]
    kw = dict(T=T, max_stride=cfg.train.max_stride, p_skip=cfg.train.p_skip, max_skip=cfg.train.max_skip)
    train_ds = ImageClipDataset(roots, samples_per_epoch=a.samples_per_epoch, seed=a.seed, flip=not a.no_flip, **kw)
    val_ds = ImageClipDataset(a.val_root, samples_per_epoch=a.val_clips, seed=12345, **kw)
    val_seqs = a.eval_seqs or sorted(val_ds.dirs)
    if main_proc:
        json.dump(dict(vars(a), cfg=dict(cfg), world=world), open(os.path.join(a.out, 'args.json'), 'w'), indent=1)
    joint = JointModule(det, model, cfg, K, T, a.min_iou, a.lambda_e2e)
    if world > 1:   # find_unused_parameters=True: en un paso algún parámetro del módulo puede no recibir gradiente (p. ej. clips sin personas); broadcast_buffers=False: BN del decoder en eval
        joint_ddp = torch.nn.parallel.DistributedDataParallel(joint, device_ids=[local] if torch.cuda.is_available() else None,
                                                              find_unused_parameters=True, broadcast_buffers=False)
    else:
        joint_ddp = joint
    random.seed(a.seed + rank); np.random.seed(a.seed + rank); torch.manual_seed(a.seed + rank)   # aleatoriedad distinta por proceso (FN inyectados, scheduled sampling)

    def extract_eval(images):          # decoder en eval (sin denoising), como en inferencia
        det.set_train(False)
        with det.autocast():
            out = det.dec(det.features(images))
        return out['pred_logits'], out['pred_boxes'], det.holder['emb']

    def fwd_eval(images):
        lg, bx, _ = extract_eval(images)
        return lg, bx

    @torch.no_grad()
    def eval_live():
        model.eval()
        tot = dict(n=0, ok=0, present=0, ok_present=0, assoc_sum=0.0, birth_sum=0.0, n_birth=0.0, cross=0, ok_cross=0, overflow=0, motion_sum=0.0, n_motion=0)
        for i in range(rank, len(val_ds), world):          # cada proceso evalúa sus clips (rank::world) y se suman los recuentos
            c = val_ds[i]
            lg, bx, em = extract_eval(c['images'].to(dev))
            q = topk_queries(lg.float(), bx.float(), em.float(), K)
            tid, _ = match_queries(matcher, q, c['gts'], dev, a.min_iou)
            batch = dict(emb=q['emb'][None].float(), boxes=q['boxes'][None], scores=q['scores'][None], tid=tid[None].to(dev),
                         frames=c['frames'][None], cross=[None])
            _, s = run_batch(model, batch, cfg, dev, False, 1.0)
            for k in tot:
                tot[k] += s[k]
        model.train(); det.set_train(True)
        tot = du.sum_dicts(tot)
        return dict(acc=tot['ok'] / max(tot['n'], 1), acc_present=tot['ok_present'] / max(tot['present'], 1),
                    loss=tot['assoc_sum'] / max(tot['n'], 1), birth_loss=tot['birth_sum'] / max(tot['n_birth'], 1), motion_nll=tot['motion_sum'] / max(tot['n_motion'], 1))

    with torch.no_grad():
        d0 = eval_deta(fwd_eval, a.val_root, val_seqs, a.deta_stride, a.deta_thr, os.path.join(a.out, 'work'), device=dev,
                       num_workers=a.num_workers, extra_thrs=a.deta_extra_thrs, dist=du)
    v0 = eval_live()
    c0 = control_value(d0, a.deta_metric)
    P(f'inicial: DetA({a.deta_metric})={c0:.2f} por umbral={ {k: round(x, 2) for k, x in d0.get("DetA_by_thr", {}).items()} } '
      f'val_acc(asignación)={v0["acc"]:.4f}', flush=True)
    best_acc, step, t_start, stop_reason = -1.0, 0, time.time(), None
    for ep in range(a.epochs):
        train_ds.seed = a.seed + ep
        ss_ep = cfg.train.ss_max * min(1.0, ep / ss_warmup) if ss_warmup else cfg.train.ss_max
        sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True, seed=a.seed) if world > 1 else None
        if sampler is not None:
            sampler.set_epoch(ep)             # cada época reparte los clips de otra forma, sin solaparse entre procesos
        dl = torch.utils.data.DataLoader(train_ds, batch_size=B, shuffle=(sampler is None), sampler=sampler, num_workers=a.num_workers,
                                         collate_fn=collate_images, drop_last=True)
        run, pr, t0 = dict(det=0.0, e2e=0.0, n=0), {}, time.time()
        for batch in dl:
            images = batch['images'].flatten(0, 1).to(dev)
            gts = [g for g_clip in batch['gts'] for g in g_clip]
            loss, loss_det, loss_e2e, loss_dict, parts = joint_ddp(images, gts, batch['frames'], ss_ep, ep, step)
            f = min(1.0, (step + 1) / max(a.warmup_steps, 1))
            f_det = 0.0 if step < det_freeze else min(1.0, (step - det_freeze + 1) / max(a.warmup_steps, 1))
            for g in opt.param_groups:
                g['lr'] = g['base'] * (f if g['kind'] == 'mod' else f_det)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(det_params, a.clip_det)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad)
            scaler.step(opt); scaler.update()
            if not torch.isfinite(loss):
                print(f'[rank {rank}] pérdida no finita; se detiene', loss_dict, float(loss_e2e), flush=True); sys.exit(1)   # torchrun termina el resto de procesos
            run['det'] += float(loss_det); run['e2e'] += float(loss_e2e); run['n'] += 1; step += 1
            for k, x in parts.items():
                pr[k] = pr.get(k, 0.0) + x
            if step % a.print_freq == 0 and main_proc:
                pstr = ' '.join(f'{k} {x:.3f}' for k, x in parts.items())
                print(f'ep {ep} step {step} det {float(loss_det):.3f} e2e {float(loss_e2e):.3f} ({pstr}) '
                      f'lr_dec {opt.param_groups[0]["lr"]:.2e} {(time.time() - t0) / run["n"]:.2f}s/it', flush=True)
        with torch.no_grad():
            d = eval_deta(fwd_eval, a.val_root, val_seqs, a.deta_stride, a.deta_thr, os.path.join(a.out, 'work'), device=dev,
                          num_workers=a.num_workers, extra_thrs=a.deta_extra_thrs, dist=du)
        v = eval_live()
        cv = control_value(d, a.deta_metric)
        ok = bool(du.broadcast_object(cv >= c0 - a.deta_tol))     # la decisión de parar la toma rank 0 y es la misma en todos: si no, unos procesos salen del bucle y otros siguen (DDP se bloquea)
        parts = {f'train_{k}': x / max(run['n'], 1) for k, x in pr.items()}                  # e2e sin ponderar, por partes
        nb_name, nm_name = val_names(cfg)                                                      # H1/H2: birth_loss, motion_nll; H3: app_loss, motion_loss
        vparts = {f'val_{nb_name}': v['birth_loss'], **({f'val_{nm_name}': v['motion_nll']} if has_motion(cfg) else {})}
        rec = dict(epoch=ep, train_det=run['det'] / max(run['n'], 1), train_e2e=run['e2e'] / max(run['n'], 1), **parts, DetA=d['DetA'], DetA_control=cv,
                   DetA_control_init=c0, DetA_by_thr=d.get('DetA_by_thr'), DetA_init_by_thr=d0.get('DetA_by_thr'), n_det=d['n_det'],
                   val_acc=v['acc'], val_acc_present=v['acc_present'], val_loss=v['loss'], **vparts, deta_ok=bool(ok), time=time.time() - t0)
        P(json.dumps(rec), flush=True)
        if a.check_sync and world > 1:      # las réplicas deben seguir idénticas tras el paso del optimizador (detector completo + módulo: no solo lo registrado en JointModule)
            sums = du.all_gather_object(du.params_checksum(det.model) + du.params_checksum(model))
            P(f'sincronización de réplicas: diferencia máxima entre procesos {max(sums) - min(sums):.3e}', flush=True)
            assert max(sums) - min(sums) < 1e-5 * max(1.0, abs(sums[0])), f'las réplicas divergen: {sums}'
        e2e_state = dict(model=model.state_dict(), cfg=dict(cfg), in_dim=init['in_dim'] if init else 256, epoch=ep, val=v)
        if main_proc:
            open(os.path.join(a.out, 'log.txt'), 'a').write(json.dumps(rec) + '\n')
            torch.save({'model': det.model.state_dict()}, os.path.join(a.out, 'detector_last.pth')); torch.save(e2e_state, os.path.join(a.out, 'e2e_last.pth'))
        if ok and v['acc'] > best_acc:
            best_acc = v['acc']
            if main_proc:
                torch.save({'model': det.model.state_dict()}, os.path.join(a.out, 'detector_best.pth')); torch.save(e2e_state, os.path.join(a.out, 'e2e_best.pth'))
        du.barrier()
        if not ok:
            kept = (f'best = mejor época anterior ({best_acc:.4f} de val_acc).' if best_acc >= 0 else
                    'NINGUNA época fue válida: no hay *_best.pth; el punto de partida sigue siendo lo mejor.')
            stop_reason = f'DetA ({a.deta_metric}) cayó a {cv:.2f} (inicial {c0:.2f}, tolerancia {a.deta_tol}) en la época {ep}. Se detiene; {kept}'
            P('PARADA:', stop_reason, flush=True)
            break
    P(f'fin. val_acc inicial {v0["acc"]:.4f} -> mejor {max(best_acc, 0.0):.4f}. {stop_reason or ""} tiempo {(time.time() - t_start) / 60:.1f} min')
    du.barrier(); du.cleanup()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--det-ckpt', required=True)
    p.add_argument('--config', default=DEFAULT_CONFIG, help='config de RT-DETRv2 del detector')
    p.add_argument('--cfg', default=os.path.join(ROOT, 'tracking', 'configs', 'e2e.yml'), help='config del módulo e2e')
    p.add_argument('-u', '--update', nargs='*', default=[], help='sobrescribe claves del módulo: appearance.alpha_app=0.2 train.p_fn=0.1')
    p.add_argument('--init-e2e', default=None, help='e2e best.pth de la etapa congelada como inicialización; sin él, el módulo parte de cero')
    p.add_argument('--dancetrack', default=os.environ.get('DANCETRACK_DIR'))
    p.add_argument('--train-roots', nargs='*', default=None)
    p.add_argument('--val-root', default=os.environ.get('DANCETRACK_DIR', '') and os.path.join(os.environ.get('DANCETRACK_DIR', ''), 'val'))
    p.add_argument('--out', required=True)
    p.add_argument('--epochs', type=int, default=40); p.add_argument('--samples-per-epoch', type=int, default=1000, help='clips por época sumando TODAS las GPUs')
    p.add_argument('--clips-per-gpu', '--clips-per-step', dest='clips_per_gpu', type=int, default=4,
                   help='clips por GPU y paso (el lote efectivo es clips-por-GPU x nº de GPUs); --clips-per-step es un alias antiguo')
    p.add_argument('--lr-scale', choices=['none', 'linear', 'sqrt'], default='none', help='reescalado del lr con el nº de GPUs (por defecto ninguno; sin validar)')
    p.add_argument('--check-sync', action='store_true', help='comprueba cada época que las réplicas DDP siguen idénticas (para pruebas)')
    p.add_argument('--lr-det', type=float, default=2e-5); p.add_argument('--lambda-e2e', type=float, default=10.0)
    p.add_argument('--lr-encoder', type=float, default=None, help='lr del encoder (por defecto = --lr-det)')
    p.add_argument('--lr-backbone', type=float, default=None, help='lr del backbone (por defecto --lr-det / 10)')
    p.add_argument('--train-encoder', action=argparse.BooleanOptionalAction, default=True, help='entrenar el encoder (por defecto sí; --no-train-encoder lo congela)')
    p.add_argument('--train-backbone', action=argparse.BooleanOptionalAction, default=True,
                   help='entrenar el backbone salvo su primera capa y sus normalizaciones, como RT-DETRv2 (por defecto sí; --no-train-backbone lo congela)')
    p.add_argument('--clip-det', type=float, default=0.1); p.add_argument('--warmup-steps', type=int, default=200)
    p.add_argument('--det-freeze-steps', type=int, default=None, help='por defecto 1000 desde cero, 0 con --init-e2e')
    p.add_argument('--min-iou', type=float, default=0.5)
    p.add_argument('--amp', choices=['fp16', 'bf16', 'none'], default='fp16'); p.add_argument('--no-flip', action='store_true')
    p.add_argument('--deta-thr', type=float, default=0.6); p.add_argument('--deta-extra-thrs', type=float, nargs='*', default=[0.5, 0.7])
    p.add_argument('--deta-stride', type=int, default=4); p.add_argument('--deta-tol', type=float, default=0.5)
    p.add_argument('--deta-metric', choices=['mean', 'thr'], default='mean', help='mean = media de los umbrales; thr = solo --deta-thr')
    p.add_argument('--eval-seqs', nargs='*', default=None); p.add_argument('--val-clips', type=int, default=100)
    p.add_argument('--num-workers', type=int, default=8); p.add_argument('--print-freq', type=int, default=20)
    p.add_argument('--seed', type=int, default=int(os.environ.get('SEED', 42)))
    p.add_argument('--device', default=None, help='ignorado: con torchrun cada proceso usa la GPU de su LOCAL_RANK')
    a = p.parse_args()
    assert a.dancetrack or a.train_roots, 'define --dancetrack o DANCETRACK_DIR'
    main(a)
