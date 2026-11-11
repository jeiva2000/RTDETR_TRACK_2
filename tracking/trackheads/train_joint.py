"""TrackHeads con el RT-DETRv2 DESCONGELADO (ajuste conjunto, 1 GPU). Reutiliza la clase Detector de tracking/joint/train.py (la del módulo e2e):
decoder, encoder y backbone entrenables (salvo la primera capa del backbone y sus normalizaciones, como RT-DETRv2), BatchNorm fijos y todo el
detector en eval salvo el decoder (denoising activo).
Cada paso: un lote de B pares (t, t+k) -> un forward del detector sobre los 2B frames -> pérdida de DETECCIÓN completa de RT-DETRv2 sobre los 2B
frames + lambda · pérdida de TrackHeads (asociación, caja, presencia, atención). El gradiente de TrackHeads llega al detector por las queries
(emb del frame t y de las candidatas de t+k) y por los mapas del encoder de t+k; cajas y scores se desconectan.
Control del detector: DetA en val (frames con stride) cada época; si cae más de --deta-tol por debajo del inicial, la época no es válida y
(salvo --no-deta-stop) se para. best = mejor val_assoc_head entre las épocas válidas. Grupos de lr: decoder --lr-det, encoder (= --lr-det),
backbone (--lr-det / 10), TrackHeads --lr; warmup lineal y coseno. Con --init (best.pth de la etapa congelada) el detector aprende desde el
primer paso; desde cero, los primeros --det-freeze-steps solo aprende TrackHeads.
Salidas en --out: detector_{best,last}.pth ({'model'}: detector completo, lo carga FrozenDetector/track.py con --det-ckpt), best.pth / last.pth
de TrackHeads (args.det_ckpt apunta al detector correspondiente: diagnose.py y track.py lo usan solos), args.json y log.txt.
Uso: python -m tracking.trackheads.train_joint --det-ckpt <detector best.pth> --init <trackheads_v3 best.pth> --out <dir>"""
import argparse, json, math, os, random, sys, time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'rtdetrv2_pytorch')); sys.path.insert(0, ROOT)
from src.core import YAMLConfig
from tracking.joint.evaluate import eval_deta
from tracking.joint.train import Detector
from tracking.trackheads.data import PairDataset, collate, seq_dirs
from tracking.trackheads.losses import trackheads_loss
from tracking.trackheads.targets import assoc_targets, build_tracks
from tracking.trackheads.train import SELECT, build_model, evaluate, loss_kw, lr_at
from tracking.video.extractor import DEFAULT_CONFIG, match_queries, topk_queries


class JointDetector:
    """Envoltorio con la interfaz de FrozenDetector (infer / detect) sobre un Detector entrenable: lo usan evaluate() de train.py y, si hace
    falta, diagnose.py / track.py con el detector en memoria. Pone el decoder en eval (sin denoising) y lo deja ahí: el bucle de
    entrenamiento vuelve a set_train(True) después de evaluar."""

    def __init__(self, det, top_k):
        self.det, self.top_k = det, top_k

    @torch.no_grad()
    def infer(self, images):
        self.det.set_train(False)
        with self.det.autocast():
            feats = self.det.features(images)
            out = self.det.dec(feats)
        q = topk_queries(out['pred_logits'].float(), out['pred_boxes'].float(), self.det.holder['emb'].float(), self.top_k)
        return q, [f.float() for f in feats]

    @torch.no_grad()
    def detect(self, images, gts, min_iou=0.5):
        q, feats = self.infer(images)
        tid, _ = match_queries(self.det.criterion.matcher, q, gts, self.det.device, min_iou)
        return q, tid.to(q['scores'].device), feats


def joint_loss(det, model, batch, a, device, ep, step):
    """Un lote de pares con el detector en modo entrenamiento -> (pérdida total, dict de partes como floats)."""
    B = batch['img_t'].shape[0]
    images = torch.cat([batch['img_t'], batch['img_tk']]).to(device)
    gts = batch['gt_t'] + batch['gt_tk']
    targets = [dict(labels=torch.zeros(len(g), dtype=torch.long, device=device), boxes=g[:, 1:5].to(device).float()) for g in gts]
    feats = det.features(images)                                        # con grafo si encoder o backbone se entrenan
    with det.autocast():
        out = det.dec(feats, targets)
    emb = det.holder['emb']
    if 'dn_meta' in out and out['dn_meta'] is not None:                 # fuera las queries de denoising
        emb = emb[:, out['dn_meta']['dn_num_split'][0]:]
    with torch.autocast(device_type=torch.device(device).type, enabled=False):
        loss_dict = det.criterion(out, targets, epoch=ep, step=step, global_step=step)
    loss_det = sum(loss_dict.values())
    q = topk_queries(out['pred_logits'].float(), out['pred_boxes'].float(), emb.float(), a.top_k)
    q['boxes'], q['scores'] = q['boxes'].detach(), q['scores'].detach()
    tid, _ = match_queries(det.criterion.matcher, q, gts, device, a.min_iou)
    tid = tid.to(device)
    tg = build_tracks({k: v[:B] for k, v in q.items()}, tid[:B], batch['gt_tk'], a.det_thr, a.max_tracks)
    out_th = model(tg['emb'], tg['prev_box'], [f[B:].float() for f in feats], tg['track_mask'])
    cand = q['scores'][B:] >= a.det_thr
    model.associate(out_th, q['emb'][B:], q['boxes'][B:], cand)
    tg['assoc_target'] = assoc_targets(tg['track_id'], tid[B:], cand)
    loss_th, parts = trackheads_loss(out_th, tg, **loss_kw(a))
    total = loss_det + a.lambda_th * loss_th
    parts = dict(loss=total.detach().item(), det=loss_det.detach().item(), th=parts['loss'],
                 **{k: v for k, v in parts.items() if k != 'loss'})
    return total, parts


def control_value(d):
    return float(np.mean(list(d['DetA_by_thr'].values()))) if 'DetA_by_thr' in d else float(d['DetA'])


def main(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = a.device if (a.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, 'args.json'), 'w'), indent=1)
    det = Detector(YAMLConfig(a.det_config, PResNet=dict(pretrained=False)), a.det_ckpt, device, a.amp,
                   train_encoder=a.train_encoder, train_backbone=a.train_backbone)
    jd = JointDetector(det, a.top_k)
    model = build_model(a).to(device)
    if a.init:
        ck = torch.load(a.init, map_location='cpu')
        assert 'w_assoc' in ck['args'], f'{a.init} no tiene Associator (anterior a trackheads_v3)'
        model.load_state_dict(ck['model'])
    model.train()
    det_freeze = a.det_freeze_steps if a.det_freeze_steps is not None else (0 if a.init else 1000)
    dec_p, enc_p, bb_p = det.params('decoder'), det.params('encoder'), det.params('backbone')
    det_params = dec_p + enc_p + bb_p
    a.lr_encoder = a.lr_det if a.lr_encoder is None else a.lr_encoder
    a.lr_backbone = a.lr_det / 10 if a.lr_backbone is None else a.lr_backbone
    print(f'TrackHeads: {"inicializado desde " + a.init if a.init else "DESDE CERO"} ({sum(p.numel() for p in model.parameters()) / 1e6:.2f}M) | detector entrenable: ' +
          ', '.join(f'{n} {sum(p.numel() for p in ps) / 1e6:.2f}M (lr {lr:.1e})' for n, ps, lr in
                    (('decoder', dec_p, a.lr_det), ('encoder', enc_p, a.lr_encoder), ('backbone', bb_p, a.lr_backbone)) if ps) +
          f' | detector congelado los primeros {det_freeze} pasos', flush=True)

    def det_groups(params, lr):      # pesos con weight decay; sesgos y normalizaciones sin él (como RT-DETR)
        return [dict(params=ps, lr=lr, weight_decay=wd, base=lr, kind='det') for ps, wd in
                (([p for p in params if p.ndim > 1], 1e-4), ([p for p in params if p.ndim <= 1], 0.0)) if ps]

    opt = torch.optim.AdamW(det_groups(dec_p, a.lr_det) + det_groups(enc_p, a.lr_encoder) + det_groups(bb_p, a.lr_backbone) +
                            [dict(params=list(model.parameters()), lr=a.lr, weight_decay=a.wd, base=a.lr, kind='th')])
    scaler = torch.cuda.amp.GradScaler(enabled=(a.amp == 'fp16' and torch.device(device).type == 'cuda'))

    tr = PairDataset(a.root, 'train', a.k_min, a.k_max, a.samples, a.seed, max_seqs=a.max_seqs)
    va = PairDataset(a.root, 'val', a.k_min, a.k_max, a.val_samples, a.seed, max_seqs=a.max_seqs)
    mk = lambda ds, shuffle: torch.utils.data.DataLoader(ds, a.batch, shuffle=shuffle, num_workers=a.num_workers, collate_fn=collate,
                                                          drop_last=shuffle, pin_memory=device != 'cpu', persistent_workers=a.num_workers > 0)
    dl_tr, dl_va = mk(tr, True), mk(va, False)
    total = a.epochs * len(dl_tr)
    val_root = os.path.join(a.root, 'val')
    val_seqs = [os.path.basename(s) for s in seq_dirs(a.root, 'val', max_seqs=a.max_seqs)]

    def fwd_eval(images):
        det.set_train(False)
        with det.autocast():
            out = det.dec(det.features(images))
        return out['pred_logits'], out['pred_boxes']

    def deta():
        with torch.no_grad():
            d = eval_deta(fwd_eval, val_root, val_seqs, a.deta_stride, a.deta_thr, os.path.join(a.out, 'work'), device=device,
                          num_workers=a.num_workers, extra_thrs=a.deta_extra_thrs)
        det.set_train(True)
        return d

    last, det_last, det_best = (os.path.join(a.out, f) for f in ('last.pth', 'detector_last.pth', 'detector_best.pth'))
    start, step, best, bad = 0, 0, -1.0, 0
    if a.resume and os.path.isfile(last):
        ck = torch.load(last, map_location='cpu')
        if ck.get('stopped'):
            print(f'el entrenamiento de {a.out} terminó ({ck["stopped"]}) en la época {ck["epoch"]}; no se entrena más', flush=True)
            return ck['best']
        model.load_state_dict(ck['model']); det.model.load_state_dict(torch.load(det_last, map_location='cpu')['model']); opt.load_state_dict(ck['opt'])
        start, step, best, bad, d0 = ck['epoch'] + 1, ck['step'], ck['best'], ck['bad'], ck['deta0']
        print(f'reanudando desde la época {start} (mejor val_assoc_head {best:.4f}, DetA inicial {control_value(d0):.2f})', flush=True)
    else:
        d0 = deta()
        v0 = evaluate(jd, model, dl_va, a, device); det.set_train(True)
        f0 = lambda x: 'n/a' if x is None else f'{x:.4f}'
        print(f'inicial: DetA {control_value(d0):.2f} por umbral {d0.get("DetA_by_thr")} | val asociación cabeza {f0(v0["assoc_head"])} '
              f'(prev_box {f0(v0["assoc_prev"])}) cruces {f0(v0["cross_head"])}', flush=True)
    c0 = control_value(d0)
    log = open(os.path.join(a.out, 'log.txt'), 'a')
    stopped = None
    for ep in range(start, a.epochs):
        tr.set_epoch(ep)
        t0, acc, n = time.time(), {}, 0
        for batch in dl_tr:
            for g in opt.param_groups:
                g['lr'] = (lr_at(step, total, a.warmup, g['base']) if g['kind'] == 'th' else
                           (0.0 if step < det_freeze else lr_at(step - det_freeze, total - det_freeze, a.warmup, g['base'])))
            loss, p = joint_loss(det, model, batch, a, device, ep, step)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(det_params, a.clip_det)
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            scaler.step(opt); scaler.update()
            if not math.isfinite(p['loss']):
                sys.exit(f'pérdida no finita en el paso {step}: {p}')
            for k, v in p.items():
                acc[k] = acc.get(k, 0.0) + v
            n += 1; step += 1
            if a.print_every and step % a.print_every == 0:
                print(f'ep {ep} paso {step}/{total} ' + ' '.join(f'{k} {v:.4f}' for k, v in p.items()) +
                      f' lr_det {opt.param_groups[0]["lr"]:.1e} {(time.time() - t0) / n:.2f}s/it', flush=True)
        rec = dict(epoch=ep, step=step, time=round(time.time() - t0, 1), **{f'train_{k}': v / max(n, 1) for k, v in acc.items()})
        d = deta()
        cv = control_value(d)
        ok = cv >= c0 - a.deta_tol
        rec.update(DetA=d['DetA'], DetA_control=cv, DetA_control_init=c0, DetA_by_thr=d.get('DetA_by_thr'), deta_ok=ok)
        v = evaluate(jd, model, dl_va, a, device); det.set_train(True)
        rec.update({f'val_{k}': val for k, val in v.items()})
        score = rec[SELECT['head']]
        score = -1.0 if score is None else score
        improved = (ok or not a.deta_stop) and score > best + a.min_delta
        if improved:
            best, bad = score, 0
            torch.save({'model': det.model.state_dict()}, det_best)
            torch.save(dict(model=model.state_dict(), args=dict(vars(a), det_ckpt=det_best), epoch=ep, val=rec), os.path.join(a.out, 'best.pth'))
        else:
            bad += 1
        rec.update(select=SELECT['head'], best_val=best, improved=improved)
        log.write(json.dumps(rec) + '\n'); log.flush()
        fmt = lambda x: 'n/a' if x is None else f'{x:.4f}'
        w = model.assoc
        print(f"ep {ep} | DetA {cv:.2f} (inicial {c0:.2f}{'' if ok else ', CAE'}) | val asociación cabeza {fmt(v['assoc_head'])} (prev_box {fmt(v['assoc_prev'])}, "
              f"apariencia {fmt(v['assoc_app'])}) cruces {fmt(v['cross_head'])} (prev_box {fmt(v['cross_prev'])}) ninguna recall {fmt(v['null_recall'])} "
              f"| pesos esc_apar {w.log_app.exp().item():.2f} masa {w.w_mass.item():.2f} iou {w.w_iou.item():.2f} "
              f"| train det {rec['train_det']:.3f} th {rec['train_th']:.3f} | {rec['time']}s{' *' if improved else ''}", flush=True)
        if not ok and a.deta_stop:
            stopped = f'DetA cayó a {cv:.2f} (inicial {c0:.2f}, tolerancia {a.deta_tol})'
        elif a.patience and bad >= a.patience:
            stopped = f'{a.patience} épocas sin mejorar {SELECT["head"]}'
        torch.save({'model': det.model.state_dict()}, det_last)
        torch.save(dict(model=model.state_dict(), args=dict(vars(a), det_ckpt=det_last), opt=opt.state_dict(), epoch=ep, step=step, best=best, bad=bad,
                        deta0=d0, stopped=stopped), last)
        if stopped:
            print(f'parada: {stopped}' + ('' if best >= 0 else '; NINGUNA época fue válida: no hay best.pth'), flush=True)
            break
    log.close()
    return best


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=os.environ.get('DANCETRACK_DIR'), help='DanceTrack con train1/, train2/, val/')
    p.add_argument('--det-ckpt', required=True, help='best.pth del detector (punto de partida)')
    p.add_argument('--det-config', default=DEFAULT_CONFIG)
    p.add_argument('--init', default=None, help='best.pth de TrackHeads de la etapa congelada (recomendado: trackheads_v3)')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--amp', choices=['fp16', 'bf16', 'none'], default='fp16', help='autocast del detector (TrackHeads va en fp32)')
    p.add_argument('--seed', type=int, default=0)
    # datos
    p.add_argument('--k-min', type=int, default=1); p.add_argument('--k-max', type=int, default=10)
    p.add_argument('--samples', type=int, default=8000, help='pares por época de train')
    p.add_argument('--val-samples', type=int, default=1000)
    p.add_argument('--max-seqs', type=int, default=0)
    p.add_argument('--batch', type=int, default=4, help='pares por paso (2B frames con pérdida de detección completa)')
    p.add_argument('--num-workers', type=int, default=6)
    # tracks (como train.py)
    p.add_argument('--top-k', type=int, default=100); p.add_argument('--min-iou', type=float, default=0.5)
    p.add_argument('--det-thr', type=float, default=0.3); p.add_argument('--max-tracks', type=int, default=64)
    p.add_argument('--cross-iou', type=float, default=0.3)
    # modelo
    p.add_argument('--d-h', type=int, default=64); p.add_argument('--rank', type=int, default=8); p.add_argument('--no-prior', action='store_true')
    # detector
    p.add_argument('--lr-det', type=float, default=2e-5, help='lr del decoder')
    p.add_argument('--lr-encoder', type=float, default=None, help='por defecto = --lr-det')
    p.add_argument('--lr-backbone', type=float, default=None, help='por defecto --lr-det / 10')
    p.add_argument('--train-encoder', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--train-backbone', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--clip-det', type=float, default=0.1)
    p.add_argument('--det-freeze-steps', type=int, default=None, help='por defecto 0 con --init, 1000 desde cero')
    p.add_argument('--lambda-th', type=float, default=10.0, help='peso de la pérdida de TrackHeads frente a la de detección')
    p.add_argument('--deta-thr', type=float, default=0.6); p.add_argument('--deta-extra-thrs', type=float, nargs='*', default=[0.5, 0.7])
    p.add_argument('--deta-stride', type=int, default=4, help='DetA sobre 1 de cada N frames de val')
    p.add_argument('--deta-tol', type=float, default=0.5, help='caída de DetA (media de umbrales) tolerada respecto al inicial')
    p.add_argument('--deta-stop', action=argparse.BooleanOptionalAction, default=True, help='--no-deta-stop: solo registra la caída')
    # optimización de TrackHeads
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=5e-5, help='lr de TrackHeads (1e-4 en la etapa congelada)')
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--warmup', type=int, default=500); p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--patience', type=int, default=5); p.add_argument('--min-delta', type=float, default=1e-3)
    p.add_argument('--resume', action='store_true'); p.add_argument('--print-every', type=int, default=100)
    # pérdida de TrackHeads (como train.py)
    p.add_argument('--w-assoc', type=float, default=2.0); p.add_argument('--w-l1', type=float, default=1.0); p.add_argument('--w-giou', type=float, default=0.5)
    p.add_argument('--w-pres', type=float, default=1.0); p.add_argument('--w-attn', type=float, default=0.5); p.add_argument('--pos-weight', type=float, default=1.0)
    return p


if __name__ == '__main__':
    a = parser().parse_args()
    if not a.root:
        sys.exit('falta --root (o DANCETRACK_DIR)')
    main(a)
