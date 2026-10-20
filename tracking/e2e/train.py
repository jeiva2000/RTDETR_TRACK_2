"""Entrenamiento del módulo e2e (H1: detector congelado, features cacheadas). Misma receta que tracking/assoc/train.py para que la
comparación sea limpia: clips de T frames con stride y saltos aleatorios (clips.py), BPTT sobre el clip, scheduled sampling con rampa,
falsos negativos inyectados y muerte tras max_age. Diferencias: estado en tensores con padding + máscaras (lote de clips en paralelo) y
cabeza de nacimiento con BCE aparte (D1).
H2 (motion.mode=gru): se añade la NLL gaussiana del movimiento (loss.w_motion) sobre los tracks presentes.
Uso: python -m tracking.e2e.train --cfg tracking/configs/e2e.yml --train-cache <dir> --val-cache <dir> --out <dir> [-u train.lr=1e-3 ...]"""
import argparse, json, os, random, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tracking.e2e.config import load_config
from tracking.e2e.losses import IGNORE, assoc_loss, birth_loss
from tracking.e2e.model import E2EModel
from tracking.e2e.motion import box_delta, motion_loss
from tracking.e2e.tokens import from_cache
from tracking.video.clips import CachedClipDataset


def collate(items):
    return dict(emb=torch.stack([c['emb'] for c in items]).float(), boxes=torch.stack([c['boxes'] for c in items]),
                scores=torch.stack([c['scores'] for c in items]), tid=torch.stack([c['track_id'] for c in items]).long(),
                frames=torch.stack([c['frames'] for c in items]).long(), cross=[c.get('cross') for c in items])


def run_batch(model, batch, cfg, device, train, ss):
    """Un lote de clips en paralelo. Devuelve (pérdida total, estadísticas)."""
    emb, boxes, scores = batch['emb'].to(device), batch['boxes'].to(device), batch['scores'].to(device)
    tid, frames = batch['tid'].to(device), batch['frames'].to(device)
    B, T, K, _ = emb.shape
    W = cfg.train.burn_in
    state = model.new_state(B, device)
    sa, na, sb, nb, sm, nm = 0.0, 0, 0.0, 0, 0.0, 0
    st = dict(n=0, ok=0, present=0, ok_present=0, ok_cross=0, cross=0, overflow=0)
    for t in range(T):
        frame = frames[:, t]
        dets = from_cache(emb[:, t], boxes[:, t], scores[:, t])
        tid_t = tid[:, t]
        if train and cfg.train.p_fn > 0:   # falsos negativos inyectados: se oculta la query de un GT
            dets = dets.drop((tid_t >= 0) & (torch.rand(B, K, device=device) < cfg.train.p_fn))
        bi = t < W                          # burn-in: llena el estado sin gradiente ni pérdida
        with torch.no_grad() if bi else torch.enable_grad():
            pred = model.predict_motion(state, frame)
            logits, birth_logit, enc = model.scores(state, dets, frame, pred)
            alive = state.alive
            eq = (tid_t[:, None, :] == state.ids[:, :, None]) & dets.valid[:, None, :] & (state.ids[:, :, None] >= 0)
            target = torch.where(eq.any(-1), eq.float().argmax(-1), torch.full_like(state.ids, K))
            use_model = torch.ones(B, dtype=torch.bool) if not train else (torch.rand(B) < ss)
            assign = model.decide(logits.detach(), alive, target=target, use_model=use_model)
            tgt_l = torch.where(alive, target, torch.full_like(target, IGNORE))
        if not bi:
            la, n_a = assoc_loss(logits, tgt_l)
            sa, na = sa + la, na + int(n_a)
            if pred is not None:   # NLL del movimiento sobre los tracks cuya detección está presente (estado previo a la actualización)
                pres_m = alive & (target < K)
                obox = torch.gather(dets.box, 1, target.clamp(max=K - 1)[..., None].expand(-1, -1, 4))
                lm, n_m = motion_loss(pred[0], pred[1], box_delta(state.box, obox), pres_m)
                sm, nm = sm + lm, nm + int(n_m)
            ok = (assign == target) & alive
            st['n'] += int(alive.sum()); st['ok'] += int(ok.sum())
            pres = alive & (target < K)
            st['present'] += int(pres.sum()); st['ok_present'] += int((ok & pres).sum())
            for b in range(B):
                cr = batch['cross'][b]
                if cr is not None and cr[t]:
                    ids_c = torch.tensor(sorted({i for p in cr[t] for i in p}), device=device)
                    m = alive[b] & torch.isin(state.ids[b], ids_c)
                    st['cross'] += int(m.sum()); st['ok_cross'] += int((ok[b] & m).sum())
        with torch.no_grad() if bi else torch.enable_grad():
            state, _ = model.update(state, assign, dets, enc, frame)
            state = state.kill(frame, cfg.state.max_age)
            known = state.ids
            new = (tid_t >= 0) & dets.valid & ~((tid_t[:, :, None] == known[:, None, :]).any(-1))   # IDs GT sin track
            if not bi:
                lb, n_b = birth_loss(birth_logit, new, dets.valid.float(), cfg.loss.birth_pos_weight)
                sb, nb = sb + lb, nb + float(n_b)
            for b in range(B):
                js = new[b].nonzero().flatten()
                if len(js):
                    state, made = state.spawn(b, tid_t[b, js], model.init_state_vec(dets.emb[b, js]), dets.box[b, js], frame[b])
                    st['overflow'] += len(js) - made
    loss = cfg.loss.w_assoc * sa / max(na, 1) + cfg.loss.w_birth * sb / max(nb, 1) + cfg.loss.w_motion * sm / max(nm, 1)
    f = lambda x: float(x.detach()) if torch.is_tensor(x) else float(x)
    st.update(assoc_sum=f(sa), birth_sum=f(sb), n_birth=nb, motion_sum=f(sm), n_motion=nm)
    return loss, st


def evaluate(model, ds, cfg, device):
    model.eval()
    tot = dict(n=0, ok=0, present=0, ok_present=0, ok_cross=0, cross=0, overflow=0, assoc_sum=0.0, birth_sum=0.0, n_birth=0.0, motion_sum=0.0, n_motion=0)
    bs = cfg.train.batch
    with torch.no_grad():
        for i in range(0, len(ds), bs):
            _, s = run_batch(model, collate([ds[j] for j in range(i, min(i + bs, len(ds)))]), cfg, device, False, 1.0)
            for k in tot:
                tot[k] += s[k]
    model.train()
    out = dict(loss=tot['assoc_sum'] / max(tot['n'], 1), acc=tot['ok'] / max(tot['n'], 1),
               acc_present=tot['ok_present'] / max(tot['present'], 1), birth_loss=tot['birth_sum'] / max(tot['n_birth'], 1))
    if tot['n_motion']:
        out['motion_nll'] = tot['motion_sum'] / tot['n_motion']
    if tot['cross']:
        out['acc_cross'] = tot['ok_cross'] / tot['cross']
    if tot['overflow']:
        out['overflow'] = tot['overflow']
    return out


def main(a):
    cfg = load_config(a.cfg, a.update)
    assert cfg.e2e.enabled, 'e2e.enabled=false: el módulo está desactivado por configuración'
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    T, dev = cfg.train, a.device
    kw = dict(T=T.T + T.burn_in, max_stride=T.max_stride, p_skip=T.p_skip, max_skip=T.max_skip, gt_root=a.gt_root, cross_iou=0.5)
    train_ds = CachedClipDataset(a.train_cache, None, samples_per_epoch=T.samples_per_epoch, seed=a.seed, **kw)
    val_ds = CachedClipDataset(a.val_cache, None, samples_per_epoch=T.val_samples, seed=12345, **kw)
    in_dim = train_ds.data[train_ds.seqs[0]]['emb'].shape[-1]
    model = E2EModel(cfg, in_dim).to(dev)
    print(f'E2EModel (motion={cfg.motion.mode}): {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parámetros', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=T.lr, weight_decay=T.weight_decay)
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(cfg=cfg, seed=a.seed), open(os.path.join(a.out, 'args.json'), 'w'), indent=1)
    best = -1.0
    for ep in range(T.epochs):
        ss = T.ss_max * min(1.0, ep / max(1, T.ss_warmup))
        train_ds.seed = a.seed + ep
        order = list(range(len(train_ds))); random.shuffle(order)
        t0, run_loss, nb = time.time(), 0.0, 0
        for b in range(0, len(order), T.batch):
            loss, _ = run_batch(model, collate([train_ds[i] for i in order[b:b + T.batch]]), cfg, dev, True, ss)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), T.clip_grad)
            opt.step(); run_loss += float(loss); nb += 1
        v = evaluate(model, val_ds, cfg, dev)
        log = dict(epoch=ep, ss=ss, train_loss=run_loss / max(nb, 1), **{f'val_{k}': x for k, x in v.items()}, time=time.time() - t0)
        print(json.dumps(log), flush=True)
        open(os.path.join(a.out, 'log.txt'), 'a').write(json.dumps(log) + '\n')
        state = dict(model=model.state_dict(), cfg=dict(cfg), in_dim=in_dim, epoch=ep, val=v)
        torch.save(state, os.path.join(a.out, 'last.pth'))
        if v['acc'] > best:
            best = v['acc']; torch.save(state, os.path.join(a.out, 'best.pth'))
    print(f'mejor val_acc (asignación propia, Hungarian) = {best:.4f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'configs', 'e2e.yml'))
    p.add_argument('-u', '--update', nargs='*', default=[], help='sobrescribe claves: train.lr=1e-3 appearance.alpha_app=0.2')
    p.add_argument('--train-cache', required=True); p.add_argument('--val-cache', required=True); p.add_argument('--out', required=True)
    p.add_argument('--gt-root', default=os.environ.get('DANCETRACK_DIR'), help='para el recuento de decisiones en cruce (opcional)')
    p.add_argument('--seed', type=int, default=int(os.environ.get('SEED', 42)))
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    main(p.parse_args())
