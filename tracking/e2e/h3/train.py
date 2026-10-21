"""Bucle de entrenamiento del H3: un lote de clips de T frames procesados secuencialmente con memoria (misma receta que tracking/e2e/train.py: BPTT sobre el clip,
scheduled sampling entre las decisiones GT y las del propio modelo, drop de detecciones) con el ruido de noise.py y las pérdidas de losses.py:
  L = L_assign + h3.loss.w_motion · L_motion + h3.loss.w_app · L_app.
Se invoca desde tracking.e2e.train.run_batch cuando h3.enabled=true, así que sirven tal cual el entrenamiento con caché (fase 1: detector congelado), el descongelado
con DDP (fase 2, train_joint.py) y la evaluación en vivo. Devuelve (pérdida, estadísticas) con las MISMAS claves que el H1/H2 para que los bucles de validación valgan:
assoc_sum = L_assign · n (así assoc_sum/n es la L_assign media), birth_sum/n_birth = L_app, motion_sum/n_motion = L_motion (los logs las renombran con h3.enabled)."""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from tracking.e2e.h3.assignment import sinkhorn_nll
from tracking.e2e.h3.losses import app_infonce, motion_loss
from tracking.e2e.h3.noise import noisy_dets
from tracking.e2e.tokens import from_cache


def run_batch_h3(model, batch, cfg, device, train, ss):
    emb, boxes, scores = batch['emb'].to(device), batch['boxes'].to(device), batch['scores'].to(device)
    tid, frames = batch['tid'].to(device), batch['frames'].to(device)
    B, T, K, _ = emb.shape
    assert cfg.train.burn_in == 0, 'el H3 no usa burn-in'
    hc = cfg.h3
    mem = model.new_memory(B, device)
    sa, na, sp, np_, sm, nm = 0.0, 0, 0.0, 0, 0.0, 0
    st = dict(n=0, ok=0, present=0, ok_present=0, ok_cross=0, cross=0, overflow=0)
    for t in range(T):
        frame, tid_t = frames[:, t], tid[:, t]
        dets = from_cache(emb[:, t], boxes[:, t], scores[:, t])
        clean = dets.box
        if train:
            dets, clean = noisy_dets(dets, tid_t, hc.noise, cfg.train.p_fn)
        out = model.forward_scores(mem, dets, frame)
        dv, tv = out['det_valid'], out['trk_valid']
        eq = (tid_t[:, :, None] == mem.ids[:, None, :]) & dv[:, :, None] & tv[:, None, :] & (tid_t[:, :, None] >= 0)
        has_trk, has_det = eq.any(2), eq.any(1)
        target_det = torch.where(has_trk, eq.float().argmax(2), torch.full_like(tid_t, mem.N))          # columna del track GT o dustbin (track nuevo / FP)
        target_trk = torch.where(has_det, eq.float().argmax(1), torch.full_like(mem.ids, K))            # detección GT del track o dustbin (no observado)
        la, n_a = sinkhorn_nll(out['logP'], target_det, target_trk, dv, tv)
        sa, na = sa + la, na + int(n_a)
        if model.use_mot:
            pres = tv & has_det
            gt_box = torch.gather(clean, 1, target_trk.clamp(max=K - 1)[..., None].expand(-1, -1, 4))
            lm, n_m = motion_loss(out['box_pred'], gt_box, pres)
            sm, nm = sm + lm, nm + int(n_m)
        if model.use_app:
            lp, n_p = app_infonce(out['a_det'], out['a_trk'], tid_t, mem.ids, dv, tv, hc.loss.tau_app)
            sp, np_ = sp + lp, np_ + int(n_p)
        with torch.no_grad():
            use_model = torch.ones(B, dtype=torch.bool) if not train else (torch.rand(B) < ss)
            assign = model.decide(out, target=target_trk, use_model=use_model)
            ok = (assign == target_trk) & tv
            pres_t = tv & (target_trk < K)
            st['n'] += int(tv.sum()); st['ok'] += int(ok.sum())
            st['present'] += int(pres_t.sum()); st['ok_present'] += int((ok & pres_t).sum())
            for b in range(B):
                cr = batch['cross'][b]
                if cr is not None and cr[t]:
                    ids_c = torch.tensor(sorted({i for p in cr[t] for i in p}), device=device)
                    m = tv[b] & torch.isin(mem.ids[b], ids_c)
                    st['cross'] += int(m.sum()); st['ok_cross'] += int((ok[b] & m).sum())
        mem, _ = model.update(mem, assign, dets, frame)
        mem = mem.kill(frame, cfg.state.max_age)
        new = (tid_t >= 0) & dv & ~((tid_t[:, :, None] == mem.ids[:, None, :]).any(-1))               # IDs GT sin track (los FP sintéticos no nacen)
        for b in range(B):
            js = new[b].nonzero().flatten()
            if len(js):
                mem, made = mem.spawn(b, tid_t[b, js], model.init_app(dets.emb[b, js]), dets.box[b, js], frame[b])
                st['overflow'] += len(js) - made
    parts = dict(assign=sa / max(na, 1))
    loss = parts['assign']
    if model.use_app:
        parts['app'] = sp / max(np_, 1); loss = loss + hc.loss.w_app * parts['app']
    if model.use_mot:
        parts['motion'] = sm / max(nm, 1); loss = loss + hc.loss.w_motion * parts['motion']
    f = lambda x: float(x.detach()) if torch.is_tensor(x) else float(x)
    st.update(assoc_sum=f(parts['assign']) * st['n'], birth_sum=f(sp), n_birth=np_, motion_sum=f(sm), n_motion=nm,
              parts={k: f(v) for k, v in parts.items()})
    return loss, st
