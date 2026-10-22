"""Diagnóstico del H3 sobre la caché de val: DÓNDE falla la asociación (no solo cuánto). Usa el ID GT que la caché guarda por query (track_id) para comparar cada decisión del
modelo con la verdad, frame a frame, en dos modos:
- closed: el modelo decide y la memoria evoluciona con SUS decisiones y nacimientos (= la inferencia). Mide además qué causa cada cambio de ID (lo que cuenta IDSW).
- teacher: la memoria se actualiza con la asignación GT (como en entrenamiento), así que cada decisión se evalúa con un estado limpio. Si aquí el modelo acierta y en closed
  no, el problema es la acumulación de errores (desajuste entre entrenamiento e inferencia); si falla también aquí, no ha aprendido a asociar.
Mide, por modo: acierto de las decisiones de los tracks con su detección presente (correcta / rechazada / enlazada a otra identidad o a un falso positivo) y ausente (falso
enlace), el top-1 de cada señal por separado (apariencia, movimiento, relacional, afinidad con gates, probabilidad de Sinkhorn), los pesos de los gates, la distribución de la
probabilidad P_ij de las parejas verdaderas y su curva frente al umbral h3.min_prob, la calidad de la caja predicha frente a la última caja observada y, en closed, las causas de
los cambios de ID y los nacimientos (identidad nueva, duplicado con el track anterior vivo, re-aparición tras morir, falso positivo).
Uso: python -m tracking.e2e.h3.diagnose run --ckpt <e2e_best.pth> --cache <features/val> --out <dir> [--modes closed teacher] [--seqs ...] [-u h3.min_prob=0.3]
     python -m tracking.e2e.h3.diagnose logs --runs <e2e_unfrozen_h1> <e2e_unfrozen_h2> <e2e_unfrozen_h3>     (resumen de los log.txt del entrenamiento)
Escribe <out>/report.json y <out>/report.md. Las pistas del final son heurísticas: orientan, no demuestran."""
import argparse, json, os, sys
from collections import Counter, defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from tracking.e2e.config import deep_update, load_config, parse_overrides
from tracking.e2e.h3.geometry import giou
from tracking.e2e.h3.loop_breakers import birth_guard, duplicate_tracks
from tracking.e2e.heads import box_iou_b
from tracking.e2e.model import build_model
from tracking.e2e.tokens import from_cache

SIGNALS = ('s_app', 's_mot', 's_rel', 's', 'logP')
SIGNAL_NAMES = {'s_app': 'apariencia', 's_mot': 'movimiento (GIoU)', 's_rel': 'relacional', 's': 'afinidad con gates', 'logP': 'probabilidad de Sinkhorn'}
THRS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


DECISION_KEYS = ('frames', 'dets_valid', 'dets_fp', 'gt_dets_valid', 'tracks_alive', 'tracks_fp_identity', 'present', 'correct', 'missed', 'missed_top1_s_ok', 'swap_to_fp',
                 'swap_to_other_id', 'absent', 'absent_ok', 'false_link', 'false_link_to_fp', 'false_link_to_other_id', 'ids_gt', 'ids_pred') + tuple(f'top1_{s}' for s in SIGNALS)


def new_stats():
    return dict(c=Counter({k: 0 for k in DECISION_KEYS}), lists=defaultdict(list), gates_sum=np.zeros(3), gates_n=0, gates_true_sum=np.zeros(3), gates_true_n=0)


def analyze_frame(a, S):
    """Decisiones de un frame (todo en numpy). a: dv (K,), alive (N,), gt_slot (N,), tid (K,), assign (N,) (K = sin detección), P (K,N), señales (K,N), gates (K,N,3), hist_len (N,)."""
    c, L = S['c'], S['lists']
    K, valid = len(a['dv']), np.where(a['dv'])[0]
    live = np.where(a['alive'])[0]
    c['frames'] += 1
    c['dets_valid'] += len(valid); c['dets_fp'] += int((a['tid'][valid] < 0).sum()); c['tracks_alive'] += len(live)
    c['gt_dets_valid'] += int((a['tid'][valid] >= 0).sum())
    if len(valid) and len(live):
        g = a['gates'][valid][:, live].reshape(-1, 3)
        S['gates_sum'] += g.sum(0); S['gates_n'] += len(g)
    for i in live:
        gt = a['gt_slot'][i]
        if gt < 0:                                                    # track nacido de un falso positivo: no tiene identidad GT
            c['tracks_fp_identity'] += 1
            continue
        js = valid[a['tid'][valid] == gt]
        asg = a['assign'][i]
        if len(js):                                                   # su detección GT está presente entre las válidas
            j = js[0]
            c['present'] += 1
            if asg == j:
                c['correct'] += 1
            elif asg == K:
                c['missed'] += 1
                c['missed_top1_s_ok'] += int(valid[a['s'][valid, i].argmax()] == j)
                L['p_missed'].append(float(a['P'][j, i]))
            elif a['tid'][asg] < 0:
                c['swap_to_fp'] += 1
            else:
                c['swap_to_other_id'] += 1
            for sig in SIGNALS:
                c[f'top1_{sig}'] += int(valid[a[sig][valid, i].argmax()] == j)
            L['p_true'].append(float(a['P'][j, i]))
            others = valid[valid != j]
            L['p_wrong_present'].append(float(a['P'][others, i].max()) if len(others) else 0.0)
            S['gates_true_sum'] += a['gates'][j, i]; S['gates_true_n'] += 1
            if a['hist_len'][i] >= 2:                                 # con historial suficiente para tener velocidad
                L['motion'].append((a['box_pred'][i], a['last'][i], a['det_box'][j]))
        else:                                                         # su detección no está (oclusión, score bajo, no emparejada)
            c['absent'] += 1
            if asg == K:
                c['absent_ok'] += 1
            else:
                c['false_link'] += 1
                c['false_link_to_fp' if a['tid'][asg] < 0 else 'false_link_to_other_id'] += 1
            L['p_max_absent'].append(float(a['P'][valid, i].max()) if len(valid) else 0.0)


@torch.no_grad()
def run_sequence(model, cfg, d, device, mode, S):
    emb, box, sc = d['emb'].to(device).float(), d['boxes'].to(device).float(), d['scores'].to(device).float()
    tid_all = d['track_id'].long().numpy()
    Nf, K = emb.shape[:2]
    mem = model.new_memory(1, device)
    gt_slot = np.full(mem.N, -1, dtype=np.int64)
    last_tid, next_id, c = {}, 1, S['c']
    c['ids_gt'] += len(set(tid_all[tid_all >= 0].tolist()))
    for t in range(Nf):
        frame = torch.tensor([t + 1], device=device)
        dets = from_cache(emb[t][None], box[t][None], sc[t][None])
        out = model.forward_scores(mem, dets, frame)
        tid, alive = tid_all[t], mem.alive[0].cpu().numpy()
        model_assign = model.decide(out)[0].cpu().numpy()
        dv = out['det_valid'][0].cpu().numpy()
        a = dict(dv=dv, alive=alive, gt_slot=gt_slot, tid=tid, assign=model_assign, P=out['logP'][0, :K, :mem.N].exp().cpu().numpy(),
                 gates=out['gates'][0].cpu().numpy(), hist_len=mem.hist_len[0].cpu().numpy(), box_pred=out['box_pred'][0].cpu().numpy(),
                 last=mem.hist[0, :, -1].cpu().numpy(), det_box=box[t].cpu().numpy(), **{k: out[k][0].cpu().numpy() for k in ('s_app', 's_mot', 's_rel', 's')})
        a['logP'] = a['P']                                            # el argmax de P y el de log P coinciden
        analyze_frame(a, S)
        if mode == 'teacher':                                         # asignación GT: la memoria nunca se contamina con errores propios
            assign = np.full(mem.N, K, dtype=np.int64)
            for i in np.where(alive & (gt_slot >= 0))[0]:
                js = np.where(dv & (tid == gt_slot[i]))[0]
                if len(js):
                    assign[i] = js[0]
        else:
            assign = model_assign
        ids_before = mem.ids[0].cpu().numpy()
        mem, matched = model.update(mem, torch.from_numpy(assign)[None].to(device), dets, frame)
        used = np.zeros(K, dtype=bool)
        events = []                                                   # (GT id, id del tracker, tipo de enlace) de los tracks asignados en este frame
        for i in np.where(matched[0].cpu().numpy())[0]:
            j = int(assign[i]); used[j] = True
            events.append((int(tid[j]), int(ids_before[i]), 'link', int(gt_slot[i])))
        mem = mem.kill(frame, cfg.state.max_age)
        if mode == 'closed':                                          # mismos mecanismos que la inferencia (apagados salvo h3.dedupe_iou / h3.birth_guard_iou)
            dup = duplicate_tracks(mem.hist[0, :, -1], mem.alive[0], mem.born[0], mem.ids[0], cfg.h3.dedupe_iou)
            c['loop_dedupe_dropped'] += int(dup.sum())
            mem = mem.drop(dup[None])
        alive_now = mem.alive[0].cpu().numpy()
        gt_slot[~alive_now] = -1
        if mode == 'closed':
            for g, trk, _, slot_gt in events:                         # cambios de ID por enlaces
                if g >= 0:
                    prev = last_tid.get(g)
                    if prev is not None and prev != trk:
                        c['idchg_swap_link' if slot_gt != g else 'idchg_switch_back'] += 1
                    last_tid[g] = trk
            guard = birth_guard(box[t], out['box_pred'][0], mem.alive[0], cfg.h3.birth_guard_iou).cpu().numpy()
            cand0 = (~used) & dv & (sc[t].cpu().numpy() >= cfg.h3.new_thr)
            c['loop_births_guarded'] += int((cand0 & guard).sum())
            cand = cand0 & ~guard
            taken = box[t][torch.from_numpy(used).to(device)]
            for j in np.argsort(-sc[t].cpu().numpy()):
                if not cand[j]:
                    continue
                if len(taken) and box_iou_b(box[t, j][None, None], taken[None])[0, 0].max() >= cfg.infer.birth_nms:
                    continue
                before = mem.alive[0].cpu().numpy()
                mem, made = mem.spawn(0, torch.tensor([next_id], device=device), model.init_app(emb[t, j][None]), box[t, j][None], frame[0])
                if not made:
                    c['births_lost_no_slot'] += 1
                    continue
                gt_slot[np.where(mem.alive[0].cpu().numpy() & ~before)[0][0]] = tid[j]
                g = int(tid[j])
                if g < 0:
                    c['births_fp'] += 1
                elif g in last_tid:
                    alive_ids = set(mem.ids[0].cpu().numpy()[mem.alive[0].cpu().numpy()].tolist())
                    c['idchg_rebirth_prev_alive' if last_tid[g] in alive_ids else 'idchg_rebirth_after_kill'] += 1
                    c['births_rebirth'] += 1
                else:
                    c['births_new_identity'] += 1
                if g >= 0:
                    last_tid[g] = next_id
                next_id += 1
                taken = torch.cat([taken, box[t, j][None]])
        else:
            tracked = set(gt_slot[mem.alive[0].cpu().numpy()].tolist())
            for j in np.where(dv & (tid >= 0))[0]:
                if int(tid[j]) in tracked:
                    continue
                before = mem.alive[0].cpu().numpy()
                mem, made = mem.spawn(0, torch.tensor([int(tid[j])], device=device), model.init_app(emb[t, j][None]), box[t, j][None], frame[0])
                if made:
                    gt_slot[np.where(mem.alive[0].cpu().numpy() & ~before)[0][0]] = tid[j]
                    tracked.add(int(tid[j]))
                else:
                    c['births_lost_no_slot'] += 1
    c['ids_pred'] += next_id - 1


def rate(n, d):
    return float(n) / d if d else None


def summarize(S, mode):
    c, L = S['c'], S['lists']
    r = dict(mode=mode, counts=dict(c))
    pres, abs_ = c['present'], c['absent']
    r['decisions_present'] = dict(n=pres, correct=rate(c['correct'], pres), missed=rate(c['missed'], pres), swap_to_other_id=rate(c['swap_to_other_id'], pres),
                                  swap_to_fp=rate(c['swap_to_fp'], pres), missed_but_top1_s_was_right=rate(c['missed_top1_s_ok'], c['missed']))
    r['decisions_absent'] = dict(n=abs_, correct=rate(c['absent_ok'], abs_), false_link=rate(c['false_link'], abs_), false_link_to_fp=rate(c['false_link_to_fp'], abs_),
                                 false_link_to_other_id=rate(c['false_link_to_other_id'], abs_))
    r['top1_present'] = {sig: rate(c[f'top1_{sig}'], pres) for sig in SIGNALS}
    if S['gates_n']:
        r['gates_mean_all_pairs'] = dict(zip(('app', 'mot', 'rel'), (S['gates_sum'] / S['gates_n']).round(4).tolist()))
    if S['gates_true_n']:
        r['gates_mean_true_pairs'] = dict(zip(('app', 'mot', 'rel'), (S['gates_true_sum'] / S['gates_true_n']).round(4).tolist()))
    pt = np.array(L['p_true'])
    if len(pt):
        r['p_true'] = dict(quantiles={str(q): float(np.quantile(pt, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}, frac_below_0_5=float((pt < 0.5).mean()))
    pm = np.array(L['p_missed'])
    if len(pm):
        r['p_true_of_missed'] = dict(median=float(np.median(pm)), frac_below_0_5=float((pm < 0.5).mean()))
    pw, pa = np.array(L['p_wrong_present']), np.array(L['p_max_absent'])
    r['threshold_curve'] = [dict(min_prob=th, true_pair_kept=float((pt >= th).mean()) if len(pt) else None, wrong_det_above=float((pw >= th).mean()) if len(pw) else None,
                                 absent_track_linked=float((pa >= th).mean()) if len(pa) else None) for th in THRS]
    if L['motion']:
        pred, last, true = (torch.tensor(np.stack(x)) for x in zip(*L['motion']))
        r['motion'] = dict(n=len(pred), l1_pred=float((pred - true).abs().sum(-1).mean()), l1_last=float((last - true).abs().sum(-1).mean()),
                           giou_pred=float(giou(pred, true).mean()), giou_last=float(giou(last, true).mean()),
                           frac_pred_better=float(((pred - true).abs().sum(-1) < (last - true).abs().sum(-1)).float().mean()))
    r['detections'] = dict(valid_per_frame=rate(c['dets_valid'], c['frames']), fp_frac=rate(c['dets_fp'], c['dets_valid']),
                           tracks_alive_per_frame=rate(c['tracks_alive'], c['frames']), gt_dets_valid_per_frame=rate(c['gt_dets_valid'], c['frames']))
    if mode == 'closed':
        chg = {k[len('idchg_'):]: v for k, v in c.items() if k.startswith('idchg_')}
        tot = sum(chg.values())
        r['id_changes'] = dict(total=tot, by_cause=chg, share={k: rate(v, tot) for k, v in chg.items()})
        r['births'] = {k[len('births_'):]: v for k, v in c.items() if k.startswith('births_')}
        r['loop_breakers'] = dict(births_guarded=c['loop_births_guarded'], dedupe_dropped=c['loop_dedupe_dropped'])
        r['ids'] = dict(predicted=c['ids_pred'], gt=c['ids_gt'], ratio=rate(c['ids_pred'], c['ids_gt']))
    return r


def hints(rep):
    """Pistas heurísticas a partir de los informes (orientan, no demuestran)."""
    h = []
    cl, te = rep.get('closed'), rep.get('teacher')
    if cl:
        dp = cl['decisions_present']
        if dp['missed'] is not None and dp['missed'] > 0.1:
            pm = cl.get('p_true_of_missed', {})
            h.append(f"En closed el {dp['missed']:.0%} de los tracks con su detección presente NO se asigna (rechazados); en el {pm.get('frac_below_0_5', 0):.0%} de esos casos P_ij < 0.5 "
                     f"y en el {dp['missed_but_top1_s_was_right'] or 0:.0%} la afinidad s sí tenía la detección correcta como mejor: señal de umbral h3.min_prob demasiado estricto.")
        ic = cl['id_changes']
        if ic['total']:
            top = max(ic['by_cause'], key=ic['by_cause'].get)
            expl = dict(rebirth_prev_alive='el track anterior seguía vivo pero no se enlazó (rechazado o tomado por otra detección) y la detección nació como ID nuevo',
                        rebirth_after_kill='el track murió por max_age antes de reaparecer (re-entrada fallida)', swap_link='un track se enlazó con la detección de otra identidad',
                        switch_back='una identidad vuelve a un track anterior tras haber tenido un duplicado')
            h.append(f"Causa principal de los cambios de ID en closed: «{top}» ({ic['share'][top]:.0%} de {ic['total']}): {expl.get(top, '')}.")
        ids = cl.get('ids')
        if ids and ids['ratio'] and ids['ratio'] > 1.5:
            h.append(f"Se crean {ids['ratio']:.1f} veces más IDs que identidades GT ({ids['predicted']} frente a {ids['gt']}): fragmentación.")
    if cl and te and cl['decisions_present']['correct'] is not None and te['decisions_present']['correct'] is not None:
        gap = te['decisions_present']['correct'] - cl['decisions_present']['correct']
        h.append(f"Acierto con tracks presentes: {te['decisions_present']['correct']:.1%} en teacher frente a {cl['decisions_present']['correct']:.1%} en closed (diferencia {gap:+.1%}): "
                 + ('el modelo decide bien con estado limpio y pierde con el suyo propio → acumulación de errores / desajuste entrenamiento–inferencia.' if gap > 0.1 and te['decisions_present']['correct'] > 0.8
                    else 'el estado limpio no cambia mucho el acierto → el problema está en la propia decisión del modelo.' if gap <= 0.1 else 'el acierto es bajo incluso con estado limpio → el modelo no ha aprendido bien a asociar.'))
    ref = te or cl
    if ref:
        t1 = ref['top1_present']
        best = max(('s_app', 's_mot', 's_rel'), key=lambda k: t1[k] or 0)
        h.append(f"Top-1 por señal ({'teacher' if te else 'closed'}): " + ', '.join(f"{SIGNAL_NAMES[k]} {t1[k]:.1%}" for k in SIGNALS if t1[k] is not None) + f". La mejor señal suelta es {SIGNAL_NAMES[best]}.")
        if t1['s'] is not None and t1[best] is not None and t1['s'] < t1[best] - 0.02:
            h.append(f"La afinidad con gates ({t1['s']:.1%}) es PEOR que la mejor señal suelta ({SIGNAL_NAMES[best]}, {t1[best]:.1%}): los gates o la mezcla están empeorando la decisión.")
        if 'motion' in ref and ref['motion']['frac_pred_better'] < 0.5:
            h.append(f"La caja predicha por el GRU solo mejora a la última caja observada en el {ref['motion']['frac_pred_better']:.0%} de los casos (GIoU {ref['motion']['giou_pred']:.2f} frente a {ref['motion']['giou_last']:.2f}): el predictor de movimiento no está ayudando.")
        g = ref.get('gates_mean_all_pairs')
        if g and max(g.values()) > 0.8:
            h.append(f"Los gates se concentran en una sola rama (pesos medios {g}): posible colapso de los gates.")
    return h


def fmt(x, pct=True):
    return '—' if x is None else (f'{x:.1%}' if pct else f'{x:.3f}')


def to_markdown(rep, meta):
    L = [f"# Diagnóstico del H3\n", f"- checkpoint: `{meta['ckpt']}`\n- caché: `{meta['cache']}` ({meta['n_seqs']} secuencias)\n- configuración h3: min_prob={meta['min_prob']}, new_thr={meta['new_thr']}, det_thr={meta['det_thr']}, max_age={meta['max_age']}\n"]
    L.append('## Pistas (heurísticas)\n' + ('\n'.join(f'- {x}' for x in hints(rep)) or '- (sin pistas)') + '\n')
    for mode in ('closed', 'teacher'):
        r = rep.get(mode)
        if not r:
            continue
        dp, da = r['decisions_present'], r['decisions_absent']
        L.append(f"## Modo {mode}" + (' (el modelo y su memoria, como en inferencia)' if mode == 'closed' else ' (memoria con la asignación GT)') + '\n')
        L.append(f"**Tracks con su detección presente** (n={dp['n']}): correcta {fmt(dp['correct'])} · rechazada (sin asignar) {fmt(dp['missed'])} · enlazada a OTRA identidad {fmt(dp['swap_to_other_id'])} · enlazada a un falso positivo {fmt(dp['swap_to_fp'])}\n")
        L.append(f"**Tracks con su detección ausente** (n={da['n']}): correcto (sin asignar) {fmt(da['correct'])} · falso enlace {fmt(da['false_link'])} (a otra identidad {fmt(da['false_link_to_other_id'])}, a un falso positivo {fmt(da['false_link_to_fp'])})\n")
        L.append('**Top-1 de cada señal** (tracks presentes; la detección GT es la de mayor valor entre las válidas): ' + ' · '.join(f"{SIGNAL_NAMES[k]} {fmt(v)}" for k, v in r['top1_present'].items()) + '\n')
        if 'gates_mean_all_pairs' in r:
            L.append(f"**Gates (peso medio)**: todos los pares {r['gates_mean_all_pairs']} · parejas verdaderas {r.get('gates_mean_true_pairs')}\n")
        if 'p_true' in r:
            q = r['p_true']['quantiles']
            L.append(f"**P_ij de la pareja verdadera**: mediana {q['0.5']:.2f} (p10 {q['0.1']:.2f}, p25 {q['0.25']:.2f}, p75 {q['0.75']:.2f}, p90 {q['0.9']:.2f}); {fmt(r['p_true']['frac_below_0_5'])} por debajo de 0.5\n")
        L.append('**Curva del umbral h3.min_prob** (en este modo): fracción de parejas verdaderas que superan el umbral frente a falsos enlaces posibles\n')
        L.append('| min_prob | parejas verdaderas que lo superan | otra detección lo supera (track presente) | track ausente con algún enlace |\n|---|---|---|---|')
        L += [f"| {t['min_prob']} | {fmt(t['true_pair_kept'])} | {fmt(t['wrong_det_above'])} | {fmt(t['absent_track_linked'])} |" for t in r['threshold_curve']]
        L.append('')
        if 'motion' in r:
            m = r['motion']
            L.append(f"**Movimiento** (n={m['n']} tracks con ≥ 2 estados): L1 predicha {m['l1_pred']:.4f} frente a última caja {m['l1_last']:.4f} · GIoU predicha {m['giou_pred']:.3f} frente a {m['giou_last']:.3f} · la predicha es mejor en {fmt(m['frac_pred_better'])}\n")
        d = r['detections']
        L.append(f"**Detecciones**: {d['valid_per_frame']:.1f} válidas por frame (GT {d['gt_dets_valid_per_frame']:.1f}, falsos positivos {fmt(d['fp_frac'])}) · tracks vivos por frame {d['tracks_alive_per_frame']:.1f}\n")
        if mode == 'closed':
            ic, b, ids = r['id_changes'], r['births'], r['ids']
            L.append(f"**IDs**: {ids['predicted']} predichos frente a {ids['gt']} en el GT (x{ids['ratio']:.2f})\n")
            L.append(f"**Cambios de ID** (total {ic['total']}): " + ' · '.join(f"{k} {v} ({fmt(ic['share'][k])})" for k, v in sorted(ic['by_cause'].items(), key=lambda kv: -kv[1])) + '\n')
            L.append('**Nacimientos**: ' + ' · '.join(f'{k} {v}' for k, v in b.items()) + '\n')
            lb = r['loop_breakers']
            if lb['births_guarded'] or lb['dedupe_dropped']:
                L.append(f"**Mecanismos contra duplicados activos**: nacimientos evitados por la guarda {lb['births_guarded']} · tracks duplicados eliminados {lb['dedupe_dropped']}\n")
    return '\n'.join(L)


def diagnose(model, cfg, caches, device, modes):
    rep = {}
    for mode in modes:
        S = new_stats()
        for k, (s, d) in enumerate(caches.items()):
            run_sequence(model, cfg, d, device, mode, S)
            print(f'[{mode}] {k + 1}/{len(caches)} {s}', flush=True)
        rep[mode] = summarize(S, mode)
    return rep


def summarize_logs(runs):
    rows = []
    for r in runs:
        f = os.path.join(r, 'log.txt')
        name = os.path.basename(r.rstrip('/')).replace('e2e_unfrozen_', '')
        if not os.path.isfile(f):
            rows.append(f'| {name} | (sin log.txt) | | | | | |')
            continue
        recs = [json.loads(l) for l in open(f)]
        ok = [x for x in recs if x.get('deta_ok', True)]                       # los logs del entrenamiento congelado (train.py) no traen deta_ok
        b = max(ok, key=lambda x: x['val_acc']) if ok else None
        stop = 'sí (DetA)' if recs and not recs[-1].get('deta_ok', True) else 'no'
        rows.append(f"| {name} | {len(recs)} | {stop} | " + (f"{b['epoch']} | {b['val_acc']:.4f} | {b.get('val_acc_present', float('nan')):.4f} | {b['val_loss']:.4f} |" if b else '(ninguna época válida) | | | |'))
    return '| variante | épocas | parada por DetA | mejor época | val_acc | val_acc_present | val_loss |\n|---|---|---|---|---|---|---|\n' + '\n'.join(rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--ckpt', required=True); r.add_argument('--cache', required=True); r.add_argument('--out', required=True)
    r.add_argument('--modes', nargs='+', default=['closed', 'teacher'], choices=['closed', 'teacher'])
    r.add_argument('--seqs', nargs='*', default=None); r.add_argument('--max-seqs', type=int, default=0)
    r.add_argument('-u', '--update', nargs='*', default=[], help='sobrescribe claves de la configuración del checkpoint: h3.min_prob=0.3 h3.new_thr=0.5')
    r.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    lg = sub.add_parser('logs'); lg.add_argument('--runs', nargs='+', required=True)
    a = p.parse_args()
    if a.cmd == 'logs':
        print(summarize_logs(a.runs))
        sys.exit(0)
    ck = torch.load(a.ckpt, map_location='cpu')
    cfg = load_config(None, ck['cfg'])
    if a.update:
        deep_update(cfg, parse_overrides(a.update))
    assert cfg.h3.enabled, 'el checkpoint no es de H3 (h3.enabled=false)'
    model = build_model(cfg, ck['in_dim']).to(a.device).eval()
    model.load_state_dict(ck['model'])
    seqs = a.seqs or sorted(f[:-3] for f in os.listdir(a.cache) if f.endswith('.pt'))
    seqs = seqs[:a.max_seqs] if a.max_seqs else seqs
    caches = {s: torch.load(os.path.join(a.cache, f'{s}.pt')) for s in seqs}
    rep = diagnose(model, cfg, caches, a.device, a.modes)
    meta = dict(ckpt=a.ckpt, cache=a.cache, n_seqs=len(seqs), min_prob=cfg.h3.min_prob, new_thr=cfg.h3.new_thr, det_thr=cfg.h3.det_thr, max_age=cfg.state.max_age)
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(meta=meta, **rep, hints=hints(rep)), open(os.path.join(a.out, 'report.json'), 'w'), indent=1)
    md = to_markdown(rep, meta)
    open(os.path.join(a.out, 'report.md'), 'w').write(md)
    print(md)
