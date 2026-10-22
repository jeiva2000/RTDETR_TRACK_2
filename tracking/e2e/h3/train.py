"""Bucle de entrenamiento del H3: un lote de clips de T frames procesados secuencialmente con memoria (BPTT sobre el clip) con el ruido de noise.py y las pérdidas de losses.py:
  L = L_assign + h3.loss.w_motion · L_motion + h3.loss.w_app · L_app + h3.birth.w · L_birth (BCE de la cabeza de nacimiento, birth_head.py: positivo = detección de una identidad GT sin track vivo).
Se invoca desde tracking.e2e.train.run_batch cuando h3.enabled=true, así que sirven tal cual el entrenamiento con caché (fase 1: detector congelado), el descongelado con DDP
(fase 2, train_joint.py) y la evaluación en vivo. Devuelve (pérdida, estadísticas) con las MISMAS claves que el H1/H2 para que los bucles de validación valgan: assoc_sum = L_assign · n
(así assoc_sum/n es la L_assign media), birth_sum/n_birth = L_app, motion_sum/n_motion = L_motion (los logs las renombran con h3.enabled).

BUCLE CERRADO (h3.closed_loop, por defecto): cada clip es de UNO de dos tipos, elegido al empezar con probabilidad `ss` (la rampa por épocas de train_joint: pocas al principio, más después):
  - clip GT (teacher): las decisiones y los nacimientos vienen del GT (un track por identidad GT sin track);
  - clip cerrado: decide el modelo y nacen tracks con la regla de inferencia (detección sin asignar con score >= h3.new_thr y NMS), así que aparecen duplicados y tracks de falsos
    positivos, la situación que en inferencia desencadenaba el bucle de duplicados y que el modelo nunca había visto al entrenar. En evaluación (train=False) todos los clips son cerrados.
Los objetivos de la asignación se definen por la IDENTIDAD GT de cada hueco (no por el id del track): si dos huecos tienen la misma identidad, la detección es del más antiguo y el otro va
al dustbin ("no observado"); un hueco de un falso positivo va siempre al dustbin. Con h3.closed_loop=false se conserva la receta anterior (nacimientos GT y scheduled sampling por frame)."""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from tracking.e2e.h3.assignment import sinkhorn_nll
from tracking.e2e.h3.losses import app_infonce, birth_loss, motion_loss
from tracking.e2e.h3.noise import noisy_dets
from tracking.e2e.heads import box_iou_b
from tracking.e2e.tokens import from_cache


def clip_targets(gt, born, tid_t, dv, tv, K, N):
    """Objetivos de la asignación por identidad GT de cada hueco. gt (B,N) identidad de cada hueco (-1 = falso positivo), born (B,N), tid_t (B,K) identidad GT de cada query, dv (B,K), tv (B,N).
    -> target_det (B,K) hueco que recibe cada detección (N = dustbin: track nuevo / falso positivo), target_trk (B,N) detección de cada hueco (K = dustbin: no observado),
    has_trk (B,K), has_det (B,N). Con varios huecos de la misma identidad (duplicados) la detección es del más antiguo (nació antes; empate: menor índice)."""
    eq = (tid_t[:, :, None] == gt[:, None, :]) & dv[:, :, None] & tv[:, None, :] & (tid_t[:, :, None] >= 0)             # (B,K,N)
    has_trk = eq.any(2)
    idx = torch.arange(N, device=gt.device)
    age = (born * (N + 1) + idx[None, :]).float()                                                                      # menor = más antiguo
    pick = torch.where(eq, -age[:, None, :], torch.full(eq.shape, float('-inf'), device=gt.device)).argmax(2)
    target_det = torch.where(has_trk, pick, torch.full_like(pick, N))
    sel = (target_det[:, :, None] == idx[None, None, :]) & has_trk[:, :, None]
    has_det = sel.any(1)
    target_trk = torch.where(has_det, sel.float().argmax(1), torch.full((gt.shape[0], N), K, dtype=torch.long, device=gt.device))
    return target_det, target_trk, has_trk, has_det


def run_batch_h3(model, batch, cfg, device, train, ss):
    emb, boxes, scores = batch['emb'].to(device), batch['boxes'].to(device), batch['scores'].to(device)
    tid, frames = batch['tid'].to(device), batch['frames'].to(device)
    B, T, K, _ = emb.shape
    assert cfg.train.burn_in == 0, 'el H3 no usa burn-in'
    hc = cfg.h3
    mem = model.new_memory(B, device)
    N = mem.N
    gt = torch.full((B, N), -1, dtype=torch.long, device=device)                    # identidad GT de cada hueco de la memoria (-1: nacido de un falso positivo)
    closed_loop = bool(hc.closed_loop)
    closed = (torch.ones(B, dtype=torch.bool) if not train else (torch.rand(B) < ss)) if closed_loop else torch.zeros(B, dtype=torch.bool)   # tipo de cada clip
    next_id, seen_gt = [1] * B, [set() for _ in range(B)]
    sa, na, sp, np_, sm, nm, sb, nb = 0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0
    st = dict(n=0, ok=0, present=0, ok_present=0, ok_cross=0, cross=0, overflow=0, births=0, births_dup=0, births_fp=0, closed_clips=int(closed.sum()))
    for t in range(T):
        frame, tid_t = frames[:, t], tid[:, t]
        dets = from_cache(emb[:, t], boxes[:, t], scores[:, t])
        clean = dets.box
        if train:
            dets, clean = noisy_dets(dets, tid_t, hc.noise, cfg.train.p_fn)
        out = model.forward_scores(mem, dets, frame)
        dv, tv = out['det_valid'], out['trk_valid']
        target_det, target_trk, has_trk, has_det = clip_targets(gt, mem.born, tid_t, dv, tv, K, N)
        la, n_a = sinkhorn_nll(out['logP'], target_det, target_trk, dv, tv)
        sa, na = sa + la, na + int(n_a)
        if model.use_mot:
            pres = tv & has_det
            gt_box = torch.gather(clean, 1, target_trk.clamp(max=K - 1)[..., None].expand(-1, -1, 4))
            lm, n_m = motion_loss(out['box_pred'], gt_box, pres)
            sm, nm = sm + lm, nm + int(n_m)
        if out['birth_logit'] is not None:                                           # cabeza de nacimiento: 1 = identidad GT sin track vivo (los falsos positivos y las identidades ya trackeadas, aunque no se asignen, son 0)
            lb, n_b = birth_loss(out['birth_logit'], (tid_t >= 0) & dv & ~has_trk, dv, hc.birth.pos_weight, hc.birth.neg_weight)
            sb, nb = sb + lb, nb + int(n_b)
        if model.use_app:
            lp, n_p = app_infonce(out['a_det'], out['a_trk'], target_det, has_trk & dv, tv, hc.loss.tau_app)
            sp, np_ = sp + lp, np_ + int(n_p)
        with torch.no_grad():
            use_model = closed if closed_loop else (torch.ones(B, dtype=torch.bool) if not train else (torch.rand(B) < ss))   # legado: mezcla por frame
            assign = model.decide(out, target=target_trk, use_model=use_model)
            ok = (assign == target_trk) & tv
            pres_t = tv & (target_trk < K)
            st['n'] += int(tv.sum()); st['ok'] += int(ok.sum())
            st['present'] += int(pres_t.sum()); st['ok_present'] += int((ok & pres_t).sum())
            for b in range(B):
                cr = batch['cross'][b]
                if cr is not None and cr[t]:
                    ids_c = torch.tensor(sorted({i for p in cr[t] for i in p}), device=device)
                    m = tv[b] & torch.isin(gt[b], ids_c)
                    st['cross'] += int(m.sum()); st['ok_cross'] += int((ok[b] & m).sum())
                seen_gt[b].update(tid_t[b][dv[b] & (tid_t[b] >= 0)].tolist())
        mem, matched = model.update(mem, assign, dets, frame)
        mem = mem.kill(frame, cfg.state.max_age)
        gt = torch.where(mem.alive, gt, torch.full_like(gt, -1))
        for b in range(B):
            if closed_loop and bool(closed[b]):                                      # clip cerrado: nacimientos con la regla de inferencia (admite duplicados y falsos positivos)
                used = torch.zeros(K, dtype=torch.bool, device=device)
                used[assign[b][matched[b]]] = True
                cand_t = (~used) & dv[b] & (dets.score[b] >= hc.new_thr)
                if out['birth_logit'] is not None:                                   # nace solo lo que la cabeza de nacimiento aprobó
                    cand_t = cand_t & (torch.sigmoid(out['birth_logit'][b].detach()) >= hc.birth.thr)
                cand = cand_t.tolist()
                taken = dets.box[b][used]
                for j in torch.argsort(dets.score[b], descending=True).tolist():
                    if not cand[j]:
                        continue
                    if len(taken) and box_iou_b(dets.box[b, j][None, None], taken[None])[0, 0].max() >= cfg.infer.birth_nms:
                        continue
                    g = int(tid_t[b, j])                                                         # tipo de nacimiento, para medir cuántos sobran
                    dup_birth = g >= 0 and bool((gt[b][mem.alive[b]] == g).any())
                    before = mem.alive[b].clone()
                    mem, made = mem.spawn(b, torch.tensor([next_id[b]], device=device), model.init_app(dets.emb[b, j][None]), dets.box[b, j][None], frame[b])
                    if made:
                        gt[b, (mem.alive[b] & ~before).nonzero().flatten()] = tid_t[b, j]
                        next_id[b] += 1; st['births'] += 1; st['births_dup'] += int(dup_birth); st['births_fp'] += int(g < 0)
                        taken = torch.cat([taken, dets.box[b, j][None]])
                    else:
                        st['overflow'] += 1
            else:                                                                    # clip GT: un track por cada identidad GT sin track
                js = ((tid_t[b] >= 0) & dv[b] & ~torch.isin(tid_t[b], gt[b][mem.alive[b]])).nonzero().flatten()
                if len(js):
                    before = mem.alive[b].clone()
                    mem, made = mem.spawn(b, tid_t[b, js], model.init_app(dets.emb[b, js]), dets.box[b, js], frame[b])
                    gt[b, (mem.alive[b] & ~before).nonzero().flatten()] = tid_t[b, js][:made]
                    st['births'] += made; st['overflow'] += len(js) - made
    parts = dict(assign=sa / max(na, 1))
    loss = parts['assign']
    if model.use_app:
        parts['app'] = sp / max(np_, 1); loss = loss + hc.loss.w_app * parts['app']
    if model.use_mot:
        parts['motion'] = sm / max(nm, 1); loss = loss + hc.loss.w_motion * parts['motion']
    if model.birth is not None:
        parts['birth'] = sb / max(nb, 1); loss = loss + hc.birth.w * parts['birth']
    f = lambda x: float(x.detach()) if torch.is_tensor(x) else float(x)
    st.update(assoc_sum=f(parts['assign']) * st['n'], birth_sum=f(sp), n_birth=np_, motion_sum=f(sm), n_motion=nm, bh_sum=f(sb), bh_n=nb, gt_ids=sum(len(x) for x in seen_gt),
              parts={k: f(v) for k, v in parts.items()})
    return loss, st
