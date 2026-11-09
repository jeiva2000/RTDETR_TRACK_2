"""Entrenamiento de TrackHeads (etapa 1) con el RT-DETRv2 congelado, en línea (los mapas del encoder no se cachean: ~8.6 MB por frame).
Cada paso: un lote de pares (t, t+k) -> UN forward del detector sobre los 2B frames -> tracks del frame t (detecciones emparejadas con el GT)
y memoria del frame t+k -> TrackHeads -> pérdida (caja + presencia + atención auxiliar). Solo se entrena TrackHeads.
Validación: pares fijos de val. Métrica principal (--select assoc, por defecto): acierto de ASOCIACIÓN, si la caja predicha apunta a la
detección correcta de t+k, frente a hacerlo con prev_box (assoc_gain), en total y en cruces (assoc.py). También: IoU de la caja frente a
prev_box (iou_gain), acierto de presencia y masa de la atención en la caja GT. best.pth = mejor métrica de val; last.pth permite reanudar.
Uso: python -m tracking.trackheads.train --det-ckpt $OUTPUT_ROOT/detector_r50vd_person/best.pth --out $OUTPUT_ROOT/trackheads/<run> [...]"""
import argparse, json, math, os, random, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tracking.trackheads.assoc import assoc_flags, assoc_sums, assoc_summary
from tracking.trackheads.data import PairDataset, collate
from tracking.trackheads.losses import batch_stats, summarize, trackheads_loss
from tracking.trackheads.model import TrackHeads
from tracking.trackheads.targets import build_tracks


SELECT = dict(assoc='val_assoc_pred', iou='val_iou')   # métrica de val que elige best.pth y controla la parada temprana


def build_model(a):
    return TrackHeads(dim=256, d_h=a.d_h, rank=a.rank, use_prior=not a.no_prior)


def run_batch(det, model, batch, a, device):
    """-> (out, tg, ctx). Un forward del detector con los frames t y t+k juntos; ctx = detecciones de t+k (para la asociación)."""
    B = batch['img_t'].shape[0]
    q, tid, feats = det.detect(torch.cat([batch['img_t'], batch['img_tk']]).to(device), batch['gt_t'] + batch['gt_tk'], a.min_iou)
    q_t = {k: v[:B] for k, v in q.items()}
    tg = build_tracks(q_t, tid[:B], batch['gt_tk'], a.det_thr, a.max_tracks)
    out = model(tg['emb'], tg['prev_box'], [f[B:] for f in feats], tg['track_mask'])
    return out, tg, dict(box_k=q['boxes'][B:], sc_k=q['scores'][B:], tid_k=tid[B:])


def loss_kw(a):
    return dict(w_l1=a.w_l1, w_giou=a.w_giou, w_pres=a.w_pres, w_attn=a.w_attn, pos_weight=a.pos_weight)


def add(acc, st):
    for k, v in st.items():
        acc[k] = acc.get(k, 0) + v


@torch.no_grad()
def evaluate(det, model, dl, a, device):
    model.eval()
    acc, parts, n = {}, {}, 0
    for batch in dl:
        out, tg, ctx = run_batch(det, model, batch, a, device)
        _, p = trackheads_loss(out, tg, **loss_kw(a))
        add(acc, batch_stats(out, tg))
        add(acc, assoc_sums(assoc_flags(out, tg, **ctx, gt_t=batch['gt_t'], gt_tk=batch['gt_tk'], det_thr=a.det_thr, cross_iou=a.cross_iou)))
        add(parts, p)
        n += 1
    model.train()
    res = summarize(acc)
    res.update(assoc_summary(acc))
    res.update({f'loss_{k}' if k != 'loss' else 'loss': v / max(n, 1) for k, v in parts.items()})
    return res


def lr_at(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def make_detector(a, device):
    from tracking.trackheads.detector import FrozenDetector
    return FrozenDetector(a.det_ckpt, a.det_config, a.top_k, device, a.amp)


def main(a, detector=None):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device(a.device if (a.device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, 'args.json'), 'w'), indent=1)
    det = detector if detector is not None else make_detector(a, device)
    model = build_model(a).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)

    tr = PairDataset(a.root, 'train', a.k_min, a.k_max, a.samples, a.seed, max_seqs=a.max_seqs)
    va = PairDataset(a.root, 'val', a.k_min, a.k_max, a.val_samples, a.seed, max_seqs=a.max_seqs)
    mk = lambda ds, shuffle: torch.utils.data.DataLoader(ds, a.batch, shuffle=shuffle, num_workers=a.num_workers, collate_fn=collate,
                                                          drop_last=shuffle, pin_memory=device.type == 'cuda', persistent_workers=a.num_workers > 0)
    dl_tr, dl_va = mk(tr, True), mk(va, False)
    total = a.epochs * len(dl_tr)

    start, step, best, bad = 0, 0, -1.0, 0
    last = os.path.join(a.out, 'last.pth')
    if a.resume and os.path.isfile(last):
        ck = torch.load(last, map_location='cpu')
        old = ck['args'].get('select', 'iou')                  # las ejecuciones anteriores a --select elegían por val_iou
        if old != a.select:
            sys.exit(f'{last} se entrenó con --select {old} y ahora es --select {a.select}: su mejor valor no es comparable. '
                     f'Use otro --out (RUN_NAME) o --select {old}.')
        if ck.get('stopped'):                                   # parada temprana: relanzar no reentrena (al completar las épocas, el bucle ya no hace nada)
            print(f'el entrenamiento de {a.out} terminó por parada temprana en la época {ck["epoch"]}; no se entrena más', flush=True)
            return ck['best']
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        start, step, best, bad = ck['epoch'] + 1, ck['step'], ck['best'], ck['bad']
        print(f'reanudando desde la época {start} (mejor {SELECT[a.select]} {best:.4f})', flush=True)

    log = open(os.path.join(a.out, 'log.txt'), 'a')
    for ep in range(start, a.epochs):
        tr.set_epoch(ep)
        t0, acc, parts, n = time.time(), {}, {}, 0
        for batch in dl_tr:
            for g in opt.param_groups:
                g['lr'] = lr_at(step, total, a.warmup, a.lr)
            out, tg, _ = run_batch(det, model, batch, a, device)
            loss, p = trackheads_loss(out, tg, **loss_kw(a))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if a.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
            add(acc, batch_stats(out, tg)); add(parts, p)
            n += 1; step += 1
            if a.print_every and step % a.print_every == 0:
                print(f'ep {ep} paso {step}/{total} ' + ' '.join(f'{k} {v:.4f}' for k, v in p.items()), flush=True)
        rec = dict(epoch=ep, step=step, lr=opt.param_groups[0]['lr'], time=round(time.time() - t0, 1))
        rec.update({f'train_{k}': v for k, v in summarize(acc).items()})
        rec.update({f'train_{k}' if k == 'loss' else f'train_loss_{k}': v / max(n, 1) for k, v in parts.items()})
        rec.update({f'val_{k}': v for k, v in evaluate(det, model, dl_va, a, device).items()})
        score = rec[SELECT[a.select]]
        score = -1.0 if score is None else score
        improved = score > best + a.min_delta
        if improved:
            best, bad = score, 0
            torch.save(dict(model=model.state_dict(), args=vars(a), epoch=ep, val=rec), os.path.join(a.out, 'best.pth'))
        else:
            bad += 1
        rec.update(select=SELECT[a.select], best_val=best, improved=improved)
        log.write(json.dumps(rec) + '\n'); log.flush()
        fmt = lambda v: 'n/a' if v is None else f'{v:.4f}'
        print(f"ep {ep} | val asociación {fmt(rec['val_assoc_pred'])} (prev_box {fmt(rec['val_assoc_prev'])}, atención {fmt(rec['val_assoc_attn'])}) "
              f"cruces {fmt(rec['val_cross_pred'])} (prev_box {fmt(rec['val_cross_prev'])}, n={rec['val_cross_n']}) | val iou {rec['val_iou']:.4f} (prev_box {rec['val_iou_prev']:.4f}, ganancia {rec['val_iou_gain']:+.4f}) "
              f"iou50 {rec['val_iou50']:.3f} pres_acc {rec['val_pres_acc']:.3f} attn_mass {rec['val_attn_mass']:.3f} "
              f"| train loss {rec['train_loss']:.4f} | {rec['time']}s{' *' if improved else ''}", flush=True)
        stop = bool(a.patience and bad >= a.patience)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), args=vars(a), epoch=ep, step=step, best=best, bad=bad, stopped=stop), last)
        if stop:
            print(f'parada temprana: {a.patience} épocas sin mejorar {SELECT[a.select]}', flush=True)
            break
    log.close()
    return best


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=os.environ.get('DANCETRACK_DIR'), help='DanceTrack con train1/, train2/, val/')
    p.add_argument('--det-ckpt', default=None, help='best.pth del detector (congelado)')
    p.add_argument('--det-config', default=None)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--amp', action='store_true', help='autocast en el forward del detector')
    p.add_argument('--seed', type=int, default=0)
    # datos
    p.add_argument('--k-min', type=int, default=1)
    p.add_argument('--k-max', type=int, default=10, help='separación máxima entre los dos frames (DanceTrack: 20 fps)')
    p.add_argument('--samples', type=int, default=8000, help='pares por época de train')
    p.add_argument('--val-samples', type=int, default=1000)
    p.add_argument('--max-seqs', type=int, default=0, help='0 = todas')
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=6)
    # tracks
    p.add_argument('--top-k', type=int, default=100)
    p.add_argument('--min-iou', type=float, default=0.5, help='IoU mínimo detección–GT para que una detección sea un track')
    p.add_argument('--det-thr', type=float, default=0.3, help='score mínimo de una detección para ser track')
    p.add_argument('--max-tracks', type=int, default=64)
    p.add_argument('--cross-iou', type=float, default=0.3, help='IoU entre cajas GT a partir del cual un track está en un cruce (métrica)')
    p.add_argument('--select', choices=list(SELECT), default='assoc', help='métrica de val para best.pth y la parada temprana')
    # modelo
    p.add_argument('--d-h', type=int, default=64)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--no-prior', action='store_true', help='sin prior gaussiano en la atención')
    # optimización
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--warmup', type=int, default=500, help='pasos de warmup lineal (luego coseno)')
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--patience', type=int, default=6, help='0 = sin parada temprana')
    p.add_argument('--min-delta', type=float, default=1e-3)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--print-every', type=int, default=100)
    # pérdida
    p.add_argument('--w-l1', type=float, default=5.0)
    p.add_argument('--w-giou', type=float, default=2.0)
    p.add_argument('--w-pres', type=float, default=1.0)
    p.add_argument('--w-attn', type=float, default=1.0, help='pérdida auxiliar de atención (0 = solo caja y presencia)')
    p.add_argument('--pos-weight', type=float, default=1.0, help='pos_weight de la BCE de presencia (<1 pesa más los ausentes)')
    return p


if __name__ == '__main__':
    a = parser().parse_args()
    if a.det_config is None:
        from tracking.video.extractor import DEFAULT_CONFIG
        a.det_config = DEFAULT_CONFIG
    if not a.det_ckpt or not a.root:
        sys.exit('faltan --det-ckpt y/o --root (o DANCETRACK_DIR)')
    main(a)
