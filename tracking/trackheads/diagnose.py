"""Diagnóstico de un checkpoint de TrackHeads sobre los pares fijos de val: ¿dónde se pierde la caja?
Por tramo de k, en total y en cruces (la caja GT del track solapa con otra identidad en t o t+k), sobre tracks presentes en t+k:
- ASOCIACIÓN (assoc.py), lo que importa: sobre tracks cuya identidad está detectada en t+k, acierto al elegir la detección de t+k por IoU
  con la caja predicha (assoc_pred), con prev_box (assoc_prev: la referencia a batir) o por masa de la atención (assoc_attn); y, desde
  trackheads_v3, con la puntuación del Associator (asoc_cabeza) y solo con su término de apariencia (asoc_apariencia).
- "Ninguna" (Associator): recall entre los tracks sin candidata de su identidad en t+k y falsas "ninguna" entre los que sí la tienen.
- IoU con el GT de t+k: prev_box (quedarse quieto), predicción y TECHO = caja del propio detector en t+k para esa identidad (si la detectó).
- Error de centro en unidades de la caja GT (|Δcx|/w + |Δcy|/h): prev_box, predicción y centro atencional (attn·xy). Si el centro atencional
  no mejora a prev_box, la atención no está localizando el desplazamiento aunque attn_mass sea alto (cajas solapadas).
- Error de tamaño |log(w/w_gt)| + |log(h/h_gt)|: prev_box y predicción. La atención solo aporta el centro; si el error está sobre todo en el
  tamaño, mejorar la atención no basta.
- Atención: masa dentro de la caja GT de t+k, masa dentro de prev_box, y masa del token más pesado.
- Presencia: fracción de ausentes, recall de ausentes (prob < 0.5) y probabilidad media en presentes y ausentes.
Uso: python -m tracking.trackheads.diagnose --ckpt $OUTPUT_ROOT/trackheads/<run>/best.pth [--det-ckpt ...] [--val-samples 2000] [--out diag.json]"""
import argparse, json, os, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tracking.trackheads.assoc import assoc_flags, gt_cross
from tracking.trackheads.data import PairDataset, collate
from tracking.trackheads.losses import attn_mass_in_box, box_iou
from tracking.trackheads.train import build_model, run_batch

BUCKETS = [(1, 2), (3, 5), (6, 10), (11, 10 ** 9)]


def center_err(b, g):
    return (b[..., 0] - g[..., 0]).abs() / g[..., 2] + (b[..., 1] - g[..., 1]).abs() / g[..., 3]


def size_err(b, g):
    return (b[..., 2] / g[..., 2]).clamp(min=1e-6).log().abs() + (b[..., 3] / g[..., 3]).clamp(min=1e-6).log().abs()


@torch.no_grad()
def diagnose(det, model, dl, a, device):
    model.eval()
    rows = {k: [] for k in ('k', 'iou_prev', 'iou_pred', 'iou_det', 'has_det', 'cerr_prev', 'cerr_pred', 'cerr_attn', 'serr_prev', 'serr_pred',
                            'mass_gt', 'mass_prev', 'peak', 'a_valid', 'a_pred', 'a_prev', 'a_attn', 'a_head', 'a_app', 'cross')}
    pres = {k: [] for k in ('present', 'prob')}
    null = dict(n=0, ok=0, false=0, valid=0)
    for batch in dl:
        out, tg, ctx = run_batch(det, model, batch, a, device)
        m = tg['track_mask']
        pres['present'].append(tg['present'][m].cpu()); pres['prob'].append(out['presence_logits'].sigmoid()[m].cpu())
        f = assoc_flags(out, tg, **ctx, gt_t=batch['gt_t'], gt_tk=batch['gt_tk'], det_thr=a.det_thr, cross_iou=a.cross_iou)
        if 'no_cand' in f:
            null['n'] += int(f['no_cand'].sum()); null['ok'] += int(f['null_ok'].sum())
            null['false'] += int(f['null_false'].sum()); null['valid'] += int(f['valid'].sum())
        pm = m & (tg['present'] > 0.5)
        if not pm.any():
            continue
        # techo: la caja del detector en t+k emparejada con la misma identidad (score >= det_thr)
        tid_k, box_k, sc_k = ctx['tid_k'], ctx['box_k'], ctx['sc_k']
        eq = (tg['track_id'][:, :, None] == tid_k[:, None, :]) & (tid_k[:, None, :] >= 0) & (sc_k[:, None, :] >= a.det_thr)
        has = eq.any(-1)
        det_box = box_k.gather(1, eq.float().argmax(-1)[..., None].expand(-1, -1, 4))
        gt, prev, pred = tg['target_box'], tg['prev_box'], out['boxes']
        c_attn = out['attn'] @ out['token_xy']
        mass_gt, _ = attn_mass_in_box(out['attn'], out['token_xy'], gt)
        mass_prev, _ = attn_mass_in_box(out['attn'], out['token_xy'], prev)
        kk = batch['k'].to(device)[:, None].expand_as(m)
        no_head = torch.zeros_like(m)           # checkpoints sin Associator
        vals = dict(k=kk, iou_prev=box_iou(prev, gt), iou_pred=box_iou(pred, gt), iou_det=box_iou(det_box, gt), has_det=has.float(),
                    cerr_prev=center_err(prev, gt), cerr_pred=center_err(pred, gt), cerr_attn=center_err(torch.cat([c_attn, gt[..., 2:]], -1), gt),
                    serr_prev=size_err(prev, gt), serr_pred=size_err(pred, gt), mass_gt=mass_gt, mass_prev=mass_prev,
                    peak=out['attn'].amax(-1), a_valid=f['valid'], a_pred=f['ok_pred'], a_prev=f['ok_prev'], a_attn=f['ok_attn'],
                    a_head=f.get("ok_head", no_head), a_app=f.get("ok_app", no_head), cross=gt_cross(tg['track_id'], batch['gt_t'], batch['gt_tk'], a.cross_iou))   # todos los presentes, no solo los detectados en t+k
        for k, v in vals.items():
            rows[k].append(v[pm].float().cpu())
    R = {k: torch.cat(v) for k, v in rows.items()}
    a_head = getattr(a, 'assoc', True)
    P = {k: torch.cat(v) for k, v in pres.items()}

    def summ(sel):
        n = int(sel.sum())
        if not n:
            return dict(n=0)
        mean = lambda k: round(float(R[k][sel].mean()), 4)
        hd = sel & (R['has_det'] > 0)
        av = sel & (R['a_valid'] > 0)
        na = int(av.sum())
        acc = lambda k: round(float(R[k][av].mean()), 4) if na else None
        head = (lambda k: acc(k)) if a_head else (lambda k: None)
        return dict(n=n, n_asoc=na, asoc_prev=acc('a_prev'), asoc_pred=acc('a_pred'), asoc_atencion=acc('a_attn'),
                    asoc_cabeza=head('a_head'), asoc_apariencia=head('a_app'), iou_prev=mean('iou_prev'), iou_pred=mean('iou_pred'),
                    iou_det_techo=round(float(R['iou_det'][hd].mean()), 4) if hd.any() else None, det_encontrada=mean('has_det'),
                    centro_prev=mean('cerr_prev'), centro_pred=mean('cerr_pred'), centro_atencion=mean('cerr_attn'),
                    tamano_prev=mean('serr_prev'), tamano_pred=mean('serr_pred'),
                    masa_en_gt=mean('mass_gt'), masa_en_prev=mean('mass_prev'), pico_atencion=mean('peak'))

    res = dict(total=summ(torch.ones_like(R['k'], dtype=torch.bool)))
    for lo, hi in BUCKETS:
        res[f'k{lo}-{hi}' if hi < 10 ** 9 else f'k>={lo}'] = summ((R['k'] >= lo) & (R['k'] <= hi))
    res['cruces'] = summ(R['cross'] > 0)
    res['sin_cruce'] = summ(R['cross'] == 0)
    ab = P['present'] < 0.5
    res['presencia'] = dict(n=len(P['present']), frac_ausentes=round(float(ab.float().mean()), 4),
                            recall_ausentes=round(float((P['prob'][ab] < 0.5).float().mean()), 4) if ab.any() else None,
                            prob_media_presentes=round(float(P['prob'][~ab].mean()), 4) if (~ab).any() else None,
                            prob_media_ausentes=round(float(P['prob'][ab].mean()), 4) if ab.any() else None)
    if a_head:
        w = model.assoc
        res['ninguna'] = dict(n_sin_candidata=null['n'], recall=round(null['ok'] / null['n'], 4) if null['n'] else None,
                              falsas=round(null['false'] / null['valid'], 4) if null['valid'] else None)
        res['pesos'] = dict(escala_apariencia=round(w.log_app.exp().item(), 3), w_masa=round(w.w_mass.item(), 3), w_iou=round(w.w_iou.item(), 3))
    return res


def show(res):
    cols = ['n', 'n_asoc', 'asoc_prev', 'asoc_cabeza', 'asoc_apariencia', 'asoc_pred', 'asoc_atencion', 'iou_prev', 'iou_pred', 'iou_det_techo', 'det_encontrada', 'centro_prev', 'centro_pred', 'centro_atencion',
            'tamano_prev', 'tamano_pred', 'masa_en_gt', 'masa_en_prev', 'pico_atencion']
    print('| tramo | ' + ' | '.join(cols) + ' |')
    print('|---' * (len(cols) + 1) + '|')
    for name, r in res.items():
        if name not in ('presencia', 'ninguna', 'pesos') and isinstance(r, dict):
            print(f'| {name} | ' + ' | '.join(str(r.get(c, '')) for c in cols) + ' |')
    print('\npresencia:', res['presencia'])
    for k in ('ninguna', 'pesos'):
        if k in res:
            print(f'{k}:', res[k])


def main(a, detector=None):
    device = torch.device(a.device if (a.device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    ck = torch.load(a.ckpt, map_location='cpu')
    ta = argparse.Namespace(**ck['args'])
    for k in ('det_ckpt', 'det_config', 'root'):              # se pueden sobrescribir (otra máquina)
        if getattr(a, k):
            setattr(ta, k, getattr(a, k))
    ta.amp = ta.amp and not a.no_amp
    ta.cross_iou = a.cross_iou                                 # los checkpoints anteriores a la métrica de asociación no lo tienen
    ta.assoc = 'w_assoc' in ck['args']                         # los anteriores al Associator (v1, v2) no lo entrenaron: no se puntúa
    model = build_model(ta).to(device)
    missing, unexpected = model.load_state_dict(ck['model'], strict=False)
    bad = [k for k in missing if not (k.startswith('assoc.') and not ta.assoc)] + list(unexpected)
    assert not bad, f'el checkpoint no encaja con el modelo: {bad[:5]}'
    if detector is None:
        from tracking.trackheads.detector import FrozenDetector
        detector = FrozenDetector(ta.det_ckpt, ta.det_config, ta.top_k, device, ta.amp)
    va = PairDataset(ta.root, 'val', ta.k_min, a.k_max or ta.k_max, a.val_samples or ta.val_samples, ta.seed, max_seqs=ta.max_seqs)
    dl = torch.utils.data.DataLoader(va, a.batch or ta.batch, num_workers=a.num_workers, collate_fn=collate)
    res = diagnose(detector, model, dl, ta, device)
    res['ckpt'], res['epoch'] = os.path.abspath(a.ckpt), ck.get('epoch')
    show(res)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), 'diagnose.json')
    json.dump(res, open(out, 'w'), indent=1)
    print('guardado en', out)
    return res


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--det-ckpt', default=None, help='por defecto, el del entrenamiento (args del checkpoint)')
    p.add_argument('--det-config', default=None)
    p.add_argument('--root', default=None)
    p.add_argument('--val-samples', type=int, default=0, help='0 = los del entrenamiento')
    p.add_argument('--k-max', type=int, default=0, help='0 = el del entrenamiento; más alto para ver cómo generaliza a k mayores')
    p.add_argument('--batch', type=int, default=0)
    p.add_argument('--num-workers', type=int, default=6)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--cross-iou', type=float, default=0.3, help='IoU entre cajas GT a partir del cual un track está en un cruce')
    p.add_argument('--out', default=None, help='por defecto, diagnose.json junto al checkpoint')
    return p


if __name__ == '__main__':
    main(parser().parse_args())
