"""Diagnóstico de un checkpoint de TrackHeads sobre los pares fijos de val: ¿dónde se pierde la caja?
Por tramo de k y en total, sobre tracks presentes en t+k:
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
from tracking.trackheads.data import PairDataset, collate
from tracking.trackheads.losses import attn_mass_in_box, box_iou
from tracking.trackheads.targets import build_tracks
from tracking.trackheads.train import build_model

BUCKETS = [(1, 2), (3, 5), (6, 10), (11, 10 ** 9)]


def center_err(b, g):
    return (b[..., 0] - g[..., 0]).abs() / g[..., 2] + (b[..., 1] - g[..., 1]).abs() / g[..., 3]


def size_err(b, g):
    return (b[..., 2] / g[..., 2]).clamp(min=1e-6).log().abs() + (b[..., 3] / g[..., 3]).clamp(min=1e-6).log().abs()


@torch.no_grad()
def diagnose(det, model, dl, a, device):
    model.eval()
    rows = {k: [] for k in ('k', 'iou_prev', 'iou_pred', 'iou_det', 'has_det', 'cerr_prev', 'cerr_pred', 'cerr_attn',
                            'serr_prev', 'serr_pred', 'mass_gt', 'mass_prev', 'peak')}
    pres = {k: [] for k in ('present', 'prob')}
    for batch in dl:
        B = batch['img_t'].shape[0]
        q, tid, feats = det.detect(torch.cat([batch['img_t'], batch['img_tk']]).to(device), batch['gt_t'] + batch['gt_tk'], a.min_iou)
        tg = build_tracks({k: v[:B] for k, v in q.items()}, tid[:B], batch['gt_tk'], a.det_thr, a.max_tracks)
        out = model(tg['emb'], tg['prev_box'], [f[B:] for f in feats], tg['track_mask'])
        m = tg['track_mask']
        pres['present'].append(tg['present'][m].cpu()); pres['prob'].append(out['presence_logits'].sigmoid()[m].cpu())
        pm = m & (tg['present'] > 0.5)
        if not pm.any():
            continue
        # techo: la caja del detector en t+k emparejada con la misma identidad (score >= det_thr)
        tid_k, box_k, sc_k = tid[B:], q['boxes'][B:], q['scores'][B:]
        eq = (tg['track_id'][:, :, None] == tid_k[:, None, :]) & (tid_k[:, None, :] >= 0) & (sc_k[:, None, :] >= a.det_thr)
        has = eq.any(-1)
        det_box = box_k.gather(1, eq.float().argmax(-1)[..., None].expand(-1, -1, 4))
        gt, prev, pred = tg['target_box'], tg['prev_box'], out['boxes']
        c_attn = out['attn'] @ out['token_xy']
        mass_gt, _ = attn_mass_in_box(out['attn'], out['token_xy'], gt)
        mass_prev, _ = attn_mass_in_box(out['attn'], out['token_xy'], prev)
        kk = batch['k'].to(device)[:, None].expand_as(m)
        vals = dict(k=kk, iou_prev=box_iou(prev, gt), iou_pred=box_iou(pred, gt), iou_det=box_iou(det_box, gt), has_det=has.float(),
                    cerr_prev=center_err(prev, gt), cerr_pred=center_err(pred, gt), cerr_attn=center_err(torch.cat([c_attn, gt[..., 2:]], -1), gt),
                    serr_prev=size_err(prev, gt), serr_pred=size_err(pred, gt), mass_gt=mass_gt, mass_prev=mass_prev,
                    peak=out['attn'].amax(-1))
        for k, v in vals.items():
            rows[k].append(v[pm].float().cpu())
    R = {k: torch.cat(v) for k, v in rows.items()}
    P = {k: torch.cat(v) for k, v in pres.items()}

    def summ(sel):
        n = int(sel.sum())
        if not n:
            return dict(n=0)
        mean = lambda k: round(float(R[k][sel].mean()), 4)
        hd = sel & (R['has_det'] > 0)
        return dict(n=n, iou_prev=mean('iou_prev'), iou_pred=mean('iou_pred'),
                    iou_det_techo=round(float(R['iou_det'][hd].mean()), 4) if hd.any() else None, det_encontrada=mean('has_det'),
                    centro_prev=mean('cerr_prev'), centro_pred=mean('cerr_pred'), centro_atencion=mean('cerr_attn'),
                    tamano_prev=mean('serr_prev'), tamano_pred=mean('serr_pred'),
                    masa_en_gt=mean('mass_gt'), masa_en_prev=mean('mass_prev'), pico_atencion=mean('peak'))

    res = dict(total=summ(torch.ones_like(R['k'], dtype=torch.bool)))
    for lo, hi in BUCKETS:
        res[f'k{lo}-{hi}' if hi < 10 ** 9 else f'k>={lo}'] = summ((R['k'] >= lo) & (R['k'] <= hi))
    ab = P['present'] < 0.5
    res['presencia'] = dict(n=len(P['present']), frac_ausentes=round(float(ab.float().mean()), 4),
                            recall_ausentes=round(float((P['prob'][ab] < 0.5).float().mean()), 4) if ab.any() else None,
                            prob_media_presentes=round(float(P['prob'][~ab].mean()), 4) if (~ab).any() else None,
                            prob_media_ausentes=round(float(P['prob'][ab].mean()), 4) if ab.any() else None)
    return res


def show(res):
    cols = ['n', 'iou_prev', 'iou_pred', 'iou_det_techo', 'det_encontrada', 'centro_prev', 'centro_pred', 'centro_atencion',
            'tamano_prev', 'tamano_pred', 'masa_en_gt', 'masa_en_prev', 'pico_atencion']
    print('| tramo | ' + ' | '.join(cols) + ' |')
    print('|---' * (len(cols) + 1) + '|')
    for name, r in res.items():
        if name != 'presencia' and isinstance(r, dict):
            print(f'| {name} | ' + ' | '.join(str(r.get(c, '')) for c in cols) + ' |')
    print('\npresencia:', res['presencia'])


def main(a, detector=None):
    device = torch.device(a.device if (a.device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    ck = torch.load(a.ckpt, map_location='cpu')
    ta = argparse.Namespace(**ck['args'])
    for k in ('det_ckpt', 'det_config', 'root'):              # se pueden sobrescribir (otra máquina)
        if getattr(a, k):
            setattr(ta, k, getattr(a, k))
    ta.amp = ta.amp and not a.no_amp
    model = build_model(ta).to(device)
    model.load_state_dict(ck['model'])
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
    p.add_argument('--out', default=None, help='por defecto, diagnose.json junto al checkpoint')
    return p


if __name__ == '__main__':
    main(parser().parse_args())
