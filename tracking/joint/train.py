"""Fase 7: fine-tuning conjunto. Solo el decoder del detector (backbone y encoder congelados, sin gradientes) + módulo de asociación.

Pérdida = pérdida de detección completa de RT-DETRv2 (incl. denoising y auxiliares) sobre cada frame del clip
          + lambda_assoc * pérdida de asociación (BPTT sobre el clip), con gradiente hacia el decoder a través de los embeddings.
Grupos de parámetros: decoder a --lr-det (por defecto 2e-5), módulo a --lr-mod (por defecto 1e-3).
Control: DetA en val (umbral --deta-thr, frames con --deta-stride) cada época; si cae más de --deta-tol puntos por debajo del
DetA inicial, se informa y se detiene. best = mejor val_acc de asignación entre las épocas que cumplen esa condición.
Salidas en --out: detector_{best,last}.pth ({'model': state_dict}, compatibles con detect_sequences/cache_features),
assoc_{best,last}.pth (mismo formato que tracking.assoc.train, compatibles con tracking.assoc.track), log.txt.
"""
import argparse, contextlib, json, os, random, sys, time
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'rtdetrv2_pytorch')); sys.path.insert(0, ROOT)
from src.core import YAMLConfig
from tracking.video.extractor import DEFAULT_CONFIG, topk_queries, match_queries
from tracking.video.clips import ImageClipDataset, collate_images
from tracking.assoc.model import AssocModel
from tracking.assoc.train import run_clip
from tracking.joint.evaluate import eval_deta, eval_assoc_live


class Detector:
    """Envoltorio del RTDETR: decoder entrenable; backbone y encoder congelados (sin gradiente) por defecto, o entrenables con train_backbone / train_encoder.
    Si se entrenan, se respeta lo que fija la configuración de RT-DETRv2 (PResNet.freeze_at=0 deja congelada la primera capa del backbone y freeze_norm sus
    normalizaciones). Todo el modelo queda en eval (BatchNorm con estadísticas fijas, dropout 0 en el encoder). Captura el embedding de la última capa."""

    def __init__(self, cfg, ckpt, device, amp, train_encoder=False, train_backbone=False):
        ck = torch.load(ckpt, map_location='cpu')
        self.model = cfg.model
        self.model.load_state_dict(ck['ema']['module'] if 'ema' in ck else ck['model'])
        self.model.to(device)
        self.criterion = cfg.criterion.to(device)
        self.train_encoder, self.train_backbone = bool(train_encoder), bool(train_backbone)
        for m, trainable in ((self.model.backbone, self.train_backbone), (self.model.encoder, self.train_encoder)):
            if not trainable:
                for p in m.parameters():
                    p.requires_grad_(False)
        self.dec = self.model.decoder
        self.amp, self.device = amp, device
        self.holder = {}
        last = self.dec.decoder.layers[self.dec.decoder.eval_idx]
        last.register_forward_hook(lambda m, i, o: self.holder.__setitem__('emb', o))
        self.set_train(True)

    def set_train(self, train):
        self.model.eval()
        self.dec.train(train)
        # BatchNorm del decoder (input_proj) siempre en eval: con lotes de frames de un mismo clip (muy correlacionados) las estadísticas
        # del lote no son las de la población, y actualizar las running_* desplaza el detector respecto a como se evalúa.
        for m in self.dec.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()

    def autocast(self):
        dt = {'fp16': torch.float16, 'bf16': torch.bfloat16}.get(self.amp)
        return torch.autocast(device_type=torch.device(self.device).type, dtype=dt, enabled=dt is not None)

    def features(self, images):
        """backbone + encoder. Sin gradiente si ambos están congelados; si alguno se entrena, el grafo se construye (si el contexto lo permite: la evaluación va en
        no_grad). Con el backbone entrenable, el gradiente atraviesa el encoder aunque este esté congelado."""
        with self.autocast():
            if self.train_backbone:
                x = self.model.backbone(images)
            else:
                with torch.no_grad():
                    x = self.model.backbone(images)
            if self.train_encoder or self.train_backbone:
                return self.model.encoder(x)
            with torch.no_grad():
                return self.model.encoder(x)

    def params(self, part):
        """Parámetros entrenables de 'backbone', 'encoder' o 'decoder'."""
        m = {'backbone': self.model.backbone, 'encoder': self.model.encoder, 'decoder': self.dec}[part]
        return [p for p in m.parameters() if p.requires_grad]

    def forward(self, images, targets=None):
        """-> out del decoder, emb (N,300,256) de las queries normales (sin las de denoising)."""
        feats = self.features(images)
        with self.autocast():
            out = self.dec(feats, targets)
        emb = self.holder['emb']
        if self.dec.training and 'dn_meta' in out:
            emb = emb[:, out['dn_meta']['dn_num_split'][0]:]
        return out, emb


def main(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = a.device
    os.makedirs(a.out, exist_ok=True)
    cfg = YAMLConfig(a.config, PResNet=dict(pretrained=False))
    det = Detector(cfg, a.det_ckpt, dev, a.amp)
    if a.assoc_ckpt:   # módulo preentrenado (Fase 6): configuración y pesos del checkpoint
        ack = torch.load(a.assoc_ckpt, map_location='cpu'); ar = argparse.Namespace(**ack['args'])
        assoc = AssocModel.from_args(ar, ack['model']['proj.0.weight'].shape[1]).to(dev)
        assoc.load_state_dict(ack['model'])
        print(f'módulo de asociación: inicializado desde {a.assoc_ckpt}', flush=True)
    else:              # módulo desde cero: configuración de la línea de comandos, pesos aleatorios
        ar = argparse.Namespace(update=a.update, dim=a.dim, hidden=a.hidden, ema_alpha=a.ema_alpha, graph_radius=a.graph_radius,
                                app_aff=a.app_aff, hist_len=a.hist_len, hist_aff=a.hist_aff, write_conf=a.write_conf,
                                write_overlap=a.write_overlap, lambda_vis=a.lambda_vis, burn_in=a.burn_in, p_fn=a.p_fn,
                                max_lost=a.max_lost, max_occ_len=10, p_occ_both=0.2, cross_iou=0.5, gt_root=a.dancetrack)
        assoc = AssocModel.from_args(ar, 256).to(dev)
        ack = dict(args=vars(ar))
        print(f'módulo de asociación: DESDE CERO ({a.update}), {sum(p.numel() for p in assoc.parameters()) / 1e6:.2f}M parámetros', flush=True)
    assoc.train()
    ar.p_cross_occ, ar.w_cross = 0.0, 1.0  # sin opciones de cruces en esta fase
    scratch = not a.assoc_ckpt
    ss_warmup = a.ss_warmup if a.ss_warmup is not None else (3 if scratch else 0)
    det_freeze = a.det_freeze_steps if a.det_freeze_steps is not None else (1000 if scratch else 0)
    matcher = det.criterion.matcher

    dec_params = [p for p in det.dec.parameters() if p.requires_grad]
    groups = [dict(params=[p for p in dec_params if p.ndim > 1], lr=a.lr_det, weight_decay=1e-4, base=a.lr_det, kind='det'),
              dict(params=[p for p in dec_params if p.ndim <= 1], lr=a.lr_det, weight_decay=0.0, base=a.lr_det, kind='det'),
              dict(params=list(assoc.parameters()), lr=a.lr_mod, weight_decay=1e-4, base=a.lr_mod, kind='mod')]
    opt = torch.optim.AdamW(groups)
    scaler = torch.cuda.amp.GradScaler(enabled=(a.amp == 'fp16' and torch.device(dev).type == 'cuda'))

    W = int(getattr(ar, 'burn_in', 0))  # frames iniciales de cada clip (solo la variante hist): llenan estado e historial sin gradiente
    print(f'variante {ar.update}; burn-in {W}; clips de {W} + {a.T} frames', flush=True)
    roots = a.train_roots or [os.path.join(a.dancetrack, 'train1'), os.path.join(a.dancetrack, 'train2')]
    train_ds = ImageClipDataset(roots, T=a.T + W, samples_per_epoch=a.samples_per_epoch, max_stride=a.max_stride, p_skip=a.p_skip,
                                max_skip=a.max_skip, seed=a.seed, flip=not a.no_flip)
    val_ds = ImageClipDataset(a.val_root, T=a.T + W, samples_per_epoch=a.val_clips, max_stride=a.max_stride, p_skip=a.p_skip,
                              max_skip=a.max_skip, seed=12345)
    val_seqs = a.eval_seqs or sorted(val_ds.dirs)
    json.dump(dict(vars(a), assoc_args=ack['args']), open(os.path.join(a.out, 'args.json'), 'w'), indent=1)

    def fwd_eval(images):  # para DetA y asignación en val: decoder en eval (sin denoising)
        det.set_train(False)
        feats = det.features(images)
        with det.autocast():
            out = det.dec(feats)
        return out['pred_logits'], out['pred_boxes']

    def extract_eval(images):
        det.set_train(False)
        feats = det.features(images)
        with det.autocast():
            out = det.dec(feats)
        return out['pred_logits'], out['pred_boxes'], det.holder['emb']

    with torch.no_grad():
        d0 = eval_deta(fwd_eval, a.val_root, val_seqs, a.deta_stride, a.deta_thr, os.path.join(a.out, 'work'), device=dev, num_workers=a.num_workers, extra_thrs=a.deta_extra_thrs)
        v0 = eval_assoc_live(extract_eval, matcher, assoc, val_ds, ar, dev, a.top_k, a.min_iou)
    det.set_train(True)
    deta0 = d0['DetA']
    print(f'inicial: DetA={deta0:.2f} (thr {a.deta_thr}, stride {a.deta_stride}, {d0["n_det"]} detecciones)  val_acc(asignación)={v0["acc"]:.4f}'
          + (f'  DetA por umbral={ {k: round(x, 2) for k, x in d0["DetA_by_thr"].items()} }' if 'DetA_by_thr' in d0 else ''), flush=True)
    best_acc, step, t_start = -1.0, 0, time.time()
    log_path = os.path.join(a.out, 'log.txt')
    stop_reason = None
    for ep in range(a.epochs):
        train_ds.seed = a.seed + ep
        ss_ep = a.ss * min(1.0, ep / ss_warmup) if ss_warmup else a.ss   # rampa de scheduled sampling si el módulo parte de cero
        dl = torch.utils.data.DataLoader(train_ds, batch_size=a.clips_per_step, shuffle=True, num_workers=a.num_workers,
                                         collate_fn=collate_images, drop_last=True)
        run = dict(det=0.0, assoc=0.0, n=0)
        t0 = time.time()
        for batch in dl:
            B, TT = batch['images'].shape[:2]
            T = TT - W
            q_burn = None
            if W:  # burn-in: extracción en modo eval y sin gradiente (como en inferencia); no entra en la pérdida de detección
                with torch.no_grad():
                    lg, bx, em = extract_eval(batch['images'][:, :W].flatten(0, 1).to(dev))
                    q_burn = topk_queries(lg.float(), bx.float(), em.float(), a.top_k)
                    gts_burn = [g for gts in batch['gts'] for g in gts[:W]]
                    tid_burn, _ = match_queries(matcher, q_burn, gts_burn, dev, a.min_iou)
                det.set_train(True)
            images = batch['images'][:, W:].flatten(0, 1).to(dev)
            gts_main = [g for gts in batch['gts'] for g in gts[W:]]
            targets = [dict(labels=torch.zeros(len(g), dtype=torch.long, device=dev), boxes=g[:, 1:5].to(dev).float()) for g in gts_main]
            out, emb = det.forward(images, targets)
            with torch.autocast(device_type=torch.device(dev).type, enabled=False):
                loss_dict = det.criterion(out, targets, epoch=ep, step=step, global_step=step)
            loss_det = sum(loss_dict.values())

            q = topk_queries(out['pred_logits'].float(), out['pred_boxes'].float(), emb.float(), a.top_k)
            tid, _ = match_queries(matcher, q, gts_main, dev, a.min_iou)
            tot, wn = 0.0, 0.0
            for b in range(B):
                sl, sb = slice(b * T, (b + 1) * T), slice(b * W, (b + 1) * W)
                cat = (lambda k, f=(lambda x: x): torch.cat([f(q_burn[k][sb]), f(q[k][sl])])) if W else (lambda k, f=(lambda x: x): f(q[k][sl]))
                clip = dict(emb=cat('emb'), boxes=cat('boxes', lambda x: x.detach()), scores=cat('scores', lambda x: x.detach()),
                            track_id=torch.cat([tid_burn[sb], tid[sl]]) if W else tid[sl], frames=batch['frames'][b],
                            present=batch['present'][b])
                l, s = run_clip(assoc, clip, ar, dev, train=True, ss=ss_ep)
                tot = tot + l; wn += s['w']
            loss_assoc = tot / max(wn, 1.0)
            loss = loss_det + a.lambda_assoc * loss_assoc

            f = min(1.0, (step + 1) / max(a.warmup_steps, 1))
            f_det = 0.0 if step < det_freeze else min(1.0, (step - det_freeze + 1) / max(a.warmup_steps, 1))   # decoder congelado al principio
            for g in opt.param_groups:
                g['lr'] = g['base'] * (f if g['kind'] == 'mod' else f_det)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(dec_params, a.clip_det)
            torch.nn.utils.clip_grad_norm_(assoc.parameters(), a.clip_mod)
            scaler.step(opt); scaler.update()
            if not torch.isfinite(loss):
                print('pérdida no finita; se detiene', {k: float(v) for k, v in loss_dict.items()}); sys.exit(1)
            run['det'] += float(loss_det); run['assoc'] += float(loss_assoc); run['n'] += 1; step += 1
            if step % a.print_freq == 0:
                print(f'ep {ep} step {step} det {float(loss_det):.3f} assoc {float(loss_assoc):.3f} lr {opt.param_groups[0]["lr"]:.2e} '
                      f'{(time.time() - t0) / run["n"]:.2f}s/it', flush=True)

        with torch.no_grad():
            d = eval_deta(fwd_eval, a.val_root, val_seqs, a.deta_stride, a.deta_thr, os.path.join(a.out, 'work'), device=dev, num_workers=a.num_workers, extra_thrs=a.deta_extra_thrs)
            v = eval_assoc_live(extract_eval, matcher, assoc, val_ds, ar, dev, a.top_k, a.min_iou)
        det.set_train(True)
        ok_deta = d['DetA'] >= deta0 - a.deta_tol
        rec = dict(epoch=ep, train_det=run['det'] / max(run['n'], 1), train_assoc=run['assoc'] / max(run['n'], 1), DetA=d['DetA'], DetA_init=deta0,
                   n_det=d['n_det'], DetA_by_thr=d.get('DetA_by_thr'), n_det_by_thr=d.get('n_det_by_thr'), DetA_init_by_thr=d0.get('DetA_by_thr'), val_acc=v['acc'], val_acc_present=v['acc_present'], val_loss=v['loss'], deta_ok=bool(ok_deta), time=time.time() - t0)
        print(json.dumps(rec), flush=True)
        with open(log_path, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        det_state = {'model': det.model.state_dict()}
        assoc_state = dict(model=assoc.state_dict(), args=ack['args'], epoch=ep, val=v)
        torch.save(det_state, os.path.join(a.out, 'detector_last.pth')); torch.save(assoc_state, os.path.join(a.out, 'assoc_last.pth'))
        if ok_deta and v['acc'] > best_acc:
            best_acc = v['acc']
            torch.save(det_state, os.path.join(a.out, 'detector_best.pth')); torch.save(assoc_state, os.path.join(a.out, 'assoc_best.pth'))
        if not ok_deta:
            kept = (f'best = mejor época anterior ({best_acc:.4f} de val_acc).' if best_acc >= 0 else
                    'NINGUNA época fue válida: no hay *_best.pth; el punto de partida (detector y módulo originales) sigue siendo lo mejor.')
            stop_reason = (f'DetA cayó a {d["DetA"]:.2f} (inicial {deta0:.2f}, tolerancia {a.deta_tol}) en la época {ep}. '
                           f'Se detiene el entrenamiento; {kept}')
            print('PARADA:', stop_reason, flush=True)
            break
    print(f'fin. val_acc inicial {v0["acc"]:.4f} -> mejor {max(best_acc, 0.0):.4f}. {stop_reason or ""}  tiempo {(time.time() - t_start) / 60:.1f} min')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--det-ckpt', required=True, help='best.pth del detector (Fase 3)')
    p.add_argument('--assoc-ckpt', default=None, help='best.pth del módulo de la Fase 6 como inicialización; SIN él el módulo se entrena desde cero con los argumentos de abajo')
    p.add_argument('--det-freeze-steps', type=int, default=None, help='pasos iniciales con el decoder a lr 0 (solo aprende el módulo); por defecto 1000 desde cero, 0 con checkpoint')
    p.add_argument('--ss-warmup', type=int, default=None, help='épocas de rampa de scheduled sampling 0 -> --ss; por defecto 3 desde cero, 0 con checkpoint')
    # módulo desde cero (se ignoran si hay --assoc-ckpt)
    p.add_argument('--update', default='hist', choices=['none', 'ema', 'gru', 'graph', 'graph2', 'hist'])
    p.add_argument('--hist-aff', choices=['none', 'max'], default='none'); p.add_argument('--hist-len', type=int, default=8)
    p.add_argument('--write-conf', type=float, default=0.5); p.add_argument('--write-overlap', type=float, default=0.5)
    p.add_argument('--lambda-vis', type=float, default=0.5); p.add_argument('--burn-in', type=int, default=10)
    p.add_argument('--dim', type=int, default=256); p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--app-aff', choices=['mlp', 'cos'], default='cos'); p.add_argument('--ema-alpha', type=float, default=0.9)
    p.add_argument('--graph-radius', type=float, default=2.0); p.add_argument('--p-fn', type=float, default=0.1)
    p.add_argument('--max-lost', type=int, default=30)
    p.add_argument('--config', default=DEFAULT_CONFIG)
    p.add_argument('--dancetrack', default=os.environ.get('DANCETRACK_DIR'))
    p.add_argument('--train-roots', nargs='*', default=None, help='carpetas con secuencias de train (por defecto <dancetrack>/train1 y train2)')
    p.add_argument('--val-root', default=os.environ.get('DANCETRACK_DIR', '') and os.path.join(os.environ.get('DANCETRACK_DIR', ''), 'val'))
    p.add_argument('--out', required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--samples-per-epoch', type=int, default=1000, help='clips por época')
    p.add_argument('--clips-per-step', type=int, default=2)
    p.add_argument('--T', type=int, default=10); p.add_argument('--max-stride', type=int, default=3)
    p.add_argument('--p-skip', type=float, default=0.2); p.add_argument('--max-skip', type=int, default=5)
    p.add_argument('--lr-det', type=float, default=2e-5); p.add_argument('--lr-mod', type=float, default=1e-3)
    p.add_argument('--lambda-assoc', type=float, default=10.0, help='peso de la pérdida de asociación frente a la de detección (sin ajustar)')
    p.add_argument('--clip-det', type=float, default=0.1); p.add_argument('--clip-mod', type=float, default=1.0)
    p.add_argument('--warmup-steps', type=int, default=200)
    p.add_argument('--ss', type=float, default=0.5, help='scheduled sampling (constante con checkpoint; con --ss-warmup, valor final de la rampa)')
    p.add_argument('--top-k', type=int, default=100); p.add_argument('--min-iou', type=float, default=0.5)
    p.add_argument('--amp', choices=['fp16', 'bf16', 'none'], default='fp16')
    p.add_argument('--no-flip', action='store_true')
    p.add_argument('--deta-thr', type=float, default=0.6); p.add_argument('--deta-stride', type=int, default=2)
    p.add_argument('--deta-tol', type=float, default=0.5, help='puntos de DetA por debajo del inicial que se toleran antes de parar')
    p.add_argument('--deta-extra-thrs', type=float, nargs='*', default=[0.5, 0.7], help='umbrales adicionales cuyo DetA solo se registra (no controla la parada)')
    p.add_argument('--eval-seqs', nargs='*', default=None); p.add_argument('--val-clips', type=int, default=100)
    p.add_argument('--num-workers', type=int, default=8); p.add_argument('--print-freq', type=int, default=20)
    p.add_argument('--seed', type=int, default=int(os.environ.get('SEED', 42)))
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    assert a.dancetrack or a.train_roots, 'define --dancetrack o DANCETRACK_DIR'
    main(a)
