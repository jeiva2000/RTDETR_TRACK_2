"""Entrenamiento del módulo de asociación (detector congelado, features cacheadas) con BPTT sobre clips.

- Nacimientos (entrenamiento): un track nace del embedding de la query emparejada con un GT ID nuevo.
- Por frame: softmax sobre las K detecciones + "no aparece", CE contra el GT (la query con el mismo track ID, o "no aparece").
- Scheduled sampling: con prob ss (rampa lineal 0 -> --ss-max) el estado se actualiza con la asignación del propio modelo
  (Hungarian) en lugar del GT.
- Inyección de falsos negativos: con prob --p-fn se oculta la query de un GT (su objetivo pasa a "no aparece").
- Frame skipping / stride aleatorio: los da el muestreador de clips (Fase 5).
- Muerte: un track sin ver más de --max-lost frames se elimina.
Casos difíciles (cruces = pares GT con IoU > --cross-iou; requiere --gt-root; todo desactivado por defecto):
  1. --p-cross-clip : fracción de clips centrados en un frame de cruce (sobremuestreo).
  2. --w-cross      : peso de la pérdida para las decisiones de tracks que están en un cruce en ese frame.
  3. --p-cross-occ  : prob. por clip de ocultar, a partir de un cruce, a una de las dos personas (o a ambas con
                      prob --p-occ-both) durante U{1..--max-occ-len} frames del clip (su objetivo pasa a "no aparece").
Con --gt-root, la validación reporta además acc_cross (precisión en decisiones de tracks en cruce).
Variante hist (requiere --gt-root): cabeza de visibilidad (visible / ocluido / fuera) supervisada con el GT con peso --lambda-vis
(visible = hay una query emparejada con ese ID; ocluido = el ID está en el GT del frame pero sin query emparejada; fuera = no está
en el GT) y --burn-in W: los clips tienen W+T frames, los W primeros solo llenan estado e historial (sin gradiente ni pérdida).
Uso: python -m tracking.assoc.train --update gru --train-cache <dir> --val-cache <dir> --out <dir>
"""
import argparse, contextlib, json, os, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tracking.video.clips import CachedClipDataset
from tracking.assoc.model import AssocModel, UPDATES
from tracking.assoc.assign import hungarian_assign
from tracking.assoc.core import Tracks, step


def run_clip(model, c, a, device, train, ss, log=None):
    E_all, D_all, sc_all = c['emb'].to(device), c['boxes'].to(device), c['scores'].to(device)
    tid, frames = c['track_id'].to(device), c['frames'].to(device)
    T, K = tid.shape
    cross = c.get('cross') or [[] for _ in range(T)]
    cross_ids = [{i for pair in cr for i in pair} for cr in cross]
    hidden = [set() for _ in range(T)]  # opción 3: oclusión simulada a partir de un cruce
    if train and a.p_cross_occ > 0 and random.random() < a.p_cross_occ:
        events = [(t, pair) for t in range(1, T) for pair in cross[t]]
        if events:
            t0, (ia, ib) = random.choice(events)
            victims = {ia, ib} if random.random() < a.p_occ_both else {random.choice((ia, ib))}
            for t in range(t0, min(T, t0 + random.randint(1, a.max_occ_len))):
                hidden[t] |= victims
    is_hist = model.update_type == 'hist'
    W = getattr(a, 'burn_in', 0)
    present = c.get('present')
    assert not is_hist or present is not None, 'hist necesita --gt-root (presencia GT para la visibilidad)'
    tr = Tracks(model.proj[0].out_features, device, hist=model.hist_len if is_hist else 0)
    vis_n, vis_ok, vis_conf = 0, 0, np.zeros(9)
    losses, wsum, n_ok, n_dec, n_ok_present, n_present, n_ok_cross, n_cross = [], 0.0, 0, 0, 0, 0, 0, 0
    for t in range(T):
        valid = torch.ones(K, dtype=torch.bool, device=device)
        if train and a.p_fn > 0:
            drop = (tid[t] >= 0) & (torch.rand(K, device=device) < a.p_fn)
            valid &= ~drop
        if hidden[t]:
            valid &= ~torch.tensor([int(i) in hidden[t] for i in tid[t].tolist()], device=device)
        if len(tr):
            col = {int(i): j for j, i in enumerate(tid[t].tolist()) if i >= 0 and valid[j]}
            target = torch.tensor([col.get(i, K) for i in tr.ids], device=device)
            use_model = (not train) or random.random() < ss
            def gt_choice(lg):  # teacher forcing: elige el GT; la confianza es la prob. del modelo en él (como en inferencia)
                idx = torch.where(target < K, target, -1)
                pr = torch.softmax(lg.detach(), 1)[torch.arange(len(idx)), target]
                return idx, torch.where(idx >= 0, pr, torch.zeros_like(pr))
            gap0 = (frames[t] - tr.last).cpu()  # frames desde que se vio cada track, antes de decidir
            bi = t < W  # burn-in: llena estado e historial sin gradiente ni pérdida
            with torch.no_grad() if bi else contextlib.nullcontext():
                logits, idx = step(model, tr, E_all[t], D_all[t], sc_all[t], frames[t],
                                   (lambda lg: hungarian_assign(lg)) if use_model else gt_choice, valid)
            if bi:
                tr.keep(frames[t] - tr.last <= a.max_lost)
                known = set(tr.ids)
                new = [j for j, i in enumerate(tid[t].tolist()) if i >= 0 and valid[j] and i not in known]
                if new:
                    j = torch.tensor(new, device=device)
                    with torch.no_grad():
                        S0, p0 = model.birth(E_all[t][j])
                    tr.add([int(tid[t, k]) for k in new], S0, D_all[t][j], frames[t], p=p0)
                continue
            in_cross = torch.tensor([i in cross_ids[t] for i in tr.ids], device=device)
            wc = a.w_cross if train else 1.0  # la pérdida de val no se pondera (comparable entre runs)
            w = torch.where(in_cross, torch.full_like(target, wc, dtype=torch.float), torch.ones_like(target, dtype=torch.float))
            losses.append((F.cross_entropy(logits, target, reduction='none') * w).sum()); wsum += float(w.sum())
            pred = torch.where(idx >= 0, idx, torch.full_like(idx, K))
            ok = pred == target
            n_ok += int(ok.sum()); n_dec += len(target)
            pres = target < K
            n_ok_present += int(ok[pres].sum()); n_present += int(pres.sum())
            n_ok_cross += int(ok[in_cross].sum()); n_cross += int(in_cross.sum())
            if log is not None:  # diagnóstico: una fila por decisión (tgt, pred, id GT de la detección elegida, cruce, gap)
                pg = torch.where(idx >= 0, tid[t][idx.clamp(min=0)], torch.full_like(idx, -2)).cpu()
                for i in range(len(target)):
                    log.append(dict(tgt=int(target[i]) if target[i] < K else -1, pred=int(idx[i]), pred_gt=int(pg[i]),
                                    own=int(tr.ids[i]), cross=bool(in_cross[i]), gap=int(gap0[i])))
            if is_hist:  # supervisión de la visibilidad: 0 visible (hay query emparejada), 1 ocluido (en el GT, sin query), 2 fuera
                lab = torch.tensor([0 if i in col else (1 if i in present[t] else 2) for i in tr.ids], device=device)
                losses.append(a.lambda_vis * F.cross_entropy(tr.vis, lab, reduction='sum'))
                pv = tr.vis.argmax(-1)
                vis_n += len(lab); vis_ok += int((pv == lab).sum())
                np.add.at(vis_conf, (lab * 3 + pv).cpu().numpy(), 1)
                if log is not None:  # diagnóstico: visibilidad real y predicha de las decisiones de este frame
                    for r, lv, pp in zip(log[-len(lab):], lab.tolist(), pv.tolist()):
                        r['vis_gt'], r['vis_pred'] = lv, pp
            tr.keep(frames[t] - tr.last <= a.max_lost)
        known = set(tr.ids)
        new = [j for j, i in enumerate(tid[t].tolist()) if i >= 0 and valid[j] and i not in known]
        if new:
            j = torch.tensor(new, device=device)
            with torch.no_grad() if t < W else contextlib.nullcontext():  # los nacimientos del burn-in tampoco llevan gradiente
                if is_hist:
                    S0, p0 = model.birth(E_all[t][j])
                    tr.add([int(tid[t, k]) for k in new], S0, D_all[t][j], frames[t], p=p0)
                else:
                    tr.add([int(tid[t, k]) for k in new], model.init_state(E_all[t][j]), D_all[t][j], frames[t])
    loss_sum = torch.stack(losses).sum() if losses else torch.zeros((), device=device, requires_grad=train)
    return loss_sum, dict(n=n_dec, w=wsum, ok=n_ok, ok_present=n_ok_present, present=n_present, ok_cross=n_ok_cross, cross=n_cross,
                           vis_n=vis_n, vis_ok=vis_ok, vis_conf=vis_conf)


def evaluate(model, ds, a, device):
    model.eval()
    tot = dict(loss=0.0, n=0, w=0.0, ok=0, ok_present=0, present=0, ok_cross=0, cross=0, vis_n=0, vis_ok=0, vis_conf=np.zeros(9))
    with torch.no_grad():
        for i in range(len(ds)):
            l, s = run_clip(model, ds[i], a, device, train=False, ss=1.0)
            tot['loss'] += float(l)
            for k in s:
                tot[k] += s[k]
    model.train()
    out = dict(loss=tot['loss'] / max(tot['w'], 1), acc=tot['ok'] / max(tot['n'], 1), acc_present=tot['ok_present'] / max(tot['present'], 1))
    if tot['cross']:
        out['acc_cross'] = tot['ok_cross'] / tot['cross']
    if tot['vis_n']:
        out['vis_acc'] = tot['vis_ok'] / tot['vis_n']
        out['vis_conf'] = [int(x) for x in tot['vis_conf']]  # filas = clase real (visible, ocluido, fuera), columnas = predicha
    return out


def main(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = a.device
    clip_kw = dict(T=a.T + a.burn_in, max_stride=a.max_stride, p_skip=a.p_skip, max_skip=a.max_skip)  # burn-in: W frames extra al inicio
    assert a.update != 'hist' or a.gt_root, 'hist requiere --gt-root (o DANCETRACK_DIR)'
    hard = a.p_cross_clip > 0 or a.w_cross != 1.0 or a.p_cross_occ > 0
    assert a.gt_root or not hard, 'las opciones de cruces requieren --gt-root (o DANCETRACK_DIR)'
    cross_kw = dict(gt_root=a.gt_root, cross_iou=a.cross_iou)
    train_ds = CachedClipDataset(a.train_cache, a.train_seqs, samples_per_epoch=a.samples_per_epoch, seed=a.seed,
                                 p_cross=a.p_cross_clip, **cross_kw, **clip_kw)
    val_ds = CachedClipDataset(a.val_cache, a.val_seqs, samples_per_epoch=a.val_samples, seed=12345, **cross_kw, **clip_kw)  # val: muestreo uniforme
    print(f'frames de cruce (IoU>{a.cross_iou}): train {len(train_ds.cross_frames)}, val {len(val_ds.cross_frames)}', flush=True)
    in_dim = train_ds.data[train_ds.seqs[0]]['emb'].shape[-1]
    model = AssocModel.from_args(a, in_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, 'args.json'), 'w'), indent=1)
    best = -1.0
    for ep in range(a.epochs):
        ss = a.ss_max * min(1.0, ep / max(1, a.ss_warmup))
        train_ds.seed = a.seed + ep  # clips distintos en cada época, reproducibles
        order = list(range(len(train_ds))); random.shuffle(order)
        t0, run_loss, run_n = time.time(), 0.0, 0
        for b in range(0, len(order), a.batch):
            opt.zero_grad()
            tot, n, wn = 0.0, 0, 0.0
            for i in order[b:b + a.batch]:
                l, s = run_clip(model, train_ds[i], a, device, train=True, ss=ss)
                tot = tot + l; n += s['n']; wn += s['w']
            if n == 0:
                continue
            loss = tot / wn
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip_grad)
            opt.step()
            run_loss += float(tot); run_n += wn
        v = evaluate(model, val_ds, a, device)
        log = dict(epoch=ep, ss=ss, train_loss=run_loss / max(run_n, 1), **{f'val_{k}': x for k, x in v.items()}, time=time.time() - t0)
        print(json.dumps(log), flush=True)
        with open(os.path.join(a.out, 'log.txt'), 'a') as f:
            f.write(json.dumps(log) + '\n')
        state = dict(model=model.state_dict(), args=vars(a), epoch=ep, val=v)
        torch.save(state, os.path.join(a.out, 'last.pth'))
        if v['acc'] > best:
            best = v['acc']; torch.save(state, os.path.join(a.out, 'best.pth'))
    print(f'mejor val_acc (asignación propia, Hungarian) = {best:.4f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--update', choices=UPDATES, required=True)
    p.add_argument('--train-cache', required=True); p.add_argument('--val-cache', required=True)
    p.add_argument('--train-seqs', nargs='*', default=None); p.add_argument('--val-seqs', nargs='*', default=None)
    p.add_argument('--out', required=True)
    p.add_argument('--T', type=int, default=10); p.add_argument('--max-stride', type=int, default=3)
    p.add_argument('--p-skip', type=float, default=0.2); p.add_argument('--max-skip', type=int, default=5)
    p.add_argument('--p-fn', type=float, default=0.1, help='prob. de ocultar la query de un GT (falso negativo inyectado)')
    p.add_argument('--ss-max', type=float, default=0.5, help='prob. máxima de scheduled sampling')
    p.add_argument('--ss-warmup', type=int, default=5, help='épocas de rampa lineal de ss')
    p.add_argument('--max-lost', type=int, default=30, help='M: frames que se mantiene un track perdido')
    p.add_argument('--gt-root', default=os.environ.get('DANCETRACK_DIR'), help='DanceTrack (train1/train2/val) para detectar cruces')
    p.add_argument('--cross-iou', type=float, default=0.5, help='IoU entre cajas GT para considerar cruce')
    p.add_argument('--p-cross-clip', type=float, default=0.0, help='[1] fracción de clips centrados en un cruce')
    p.add_argument('--w-cross', type=float, default=1.0, help='[2] peso de la pérdida en decisiones de tracks en cruce')
    p.add_argument('--p-cross-occ', type=float, default=0.0, help='[3] prob. por clip de ocultar a una persona tras un cruce')
    p.add_argument('--max-occ-len', type=int, default=10, help='[3] duración máxima de la oclusión (frames del clip)')
    p.add_argument('--p-occ-both', type=float, default=0.2, help='[3] prob. de ocultar a las dos personas del cruce')
    p.add_argument('--dim', type=int, default=256); p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--app-aff', choices=['mlp', 'cos'], default='cos', help='rama de apariencia de la afinidad')
    p.add_argument('--hist-len', type=int, default=8, help='[hist] H: entradas del historial por track')
    p.add_argument('--hist-aff', choices=['none', 'max'], default='none', help='[hist] none = afinidad A (solo apariencia actualizada); max = B (+coseno máximo contra el historial)')
    p.add_argument('--write-conf', type=float, default=0.5, help='[hist] confianza mínima del match para escribir en el historial')
    p.add_argument('--write-overlap', type=float, default=0.5, help='[hist] solapamiento máximo con otros tracks para escribir')
    p.add_argument('--lambda-vis', type=float, default=0.5, help='[hist] peso de la pérdida de visibilidad')
    p.add_argument('--burn-in', type=int, default=0, help='[hist] frames extra al inicio del clip sin gradiente ni pérdida (llenan el historial)')
    p.add_argument('--ema-alpha', type=float, default=0.9); p.add_argument('--graph-radius', type=float, default=2.0)
    p.add_argument('--epochs', type=int, default=20); p.add_argument('--samples-per-epoch', type=int, default=2000)
    p.add_argument('--val-samples', type=int, default=300); p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--clip-grad', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=int(os.environ.get('SEED', 42)))
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    main(p.parse_args())
