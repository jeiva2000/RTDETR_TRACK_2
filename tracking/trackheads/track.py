"""Tracker en línea con TrackHeads sobre secuencias completas (DanceTrack val), para evaluar con TrackEval (HOTA, AssA, IDF1...)
con el mismo protocolo que el baseline (tracking/baseline/evaluate.py: global y subconjunto de cruces).

Un forward del detector congelado por frame (en lotes) da las queries (emb, caja, score) y los mapas del encoder. Varias configuraciones
(--configs, "nombre=modo[,clave=valor...]") comparten esas detecciones; cada una es un tracker independiente que escribe <out>/<nombre>/.
Modos (puntuación de asociación):
- head: TrackHeads + Associator. Cada track vivo genera su cabeza a partir de (emb, caja) de su última detección asignada y puntúa las
  candidatas del frame actual (score >= --det-thr) y "ninguna"; puntuación = log-softmax de assoc_logits.
- iou: la misma maquinaria con puntuación = IoU(caja de la última detección, candidata) y "ninguna" = --iou-min. Es la referencia que
  aísla lo que aporta la cabeza (mismo detector, nacimientos y muertes).
Asignación: Hungarian global (cada candidata a un track como mucho; cada track puede quedarse en "ninguna" con su coste propio).
Tracks asignados: emb y caja pasan a ser los de la detección asignada (como en el entrenamiento, el track es la detección de su último frame).
Nacimientos: candidatas sin asignar con score >= birth_thresh que no solapan (IoU < birth_nms) con las asignadas ni con otros
nacimientos (NMS voraz por score). Muerte: más de max_lost frames sin asignación.
Gestión del ciclo de vida (claves de cada configuración; por defecto, las opciones de la línea de comandos):
- min_hits: un track se CONFIRMA al acumular min_hits asignaciones seguidas (o en los primeros min_hits frames de la secuencia, como
  OC-SORT) y desde entonces sigue confirmado. Solo se escriben tracks confirmados; uno sin confirmar muere en cuanto falla un frame.
  min_hits=1: todo track está confirmado desde que nace (el comportamiento de la primera versión).
- out_thresh: no se escriben las cajas con score menor (el track sigue vivo y asignado: la caja dudosa solo no sale en el resultado).
- motion=1: velocidad constante del centro. La velocidad se estima con una media exponencial (motion_ema) de los desplazamientos entre
  asignaciones; la caja que se usa como prev_box (cabeza) o para el IoU es la última caja con el centro extrapolado
  min(frames sin ver, motion_max) frames. El tamaño no se extrapola.
- emb_update: cómo se actualiza el emb del track al asignarle una detección (la caja siempre pasa a ser la de la detección).
  * last: se reemplaza por el de la detección (la primera versión).
  * ema: media exponencial, emb = emb_ema·emb + (1 − emb_ema)·emb_detección.
  * clean: se reemplaza solo si la caja asignada no solapa (IoU < clean_iou) con ninguna otra candidata del frame. En un cruce, la
    query de una caja que cubre a dos personas mezcla a ambas; así el track conserva la apariencia de antes del cruce.
  * clean_ema: la media exponencial, solo con detecciones limpias.
  La cabeza se entrenó con el emb de UNA detección; ema y clean_ema le dan una media de varias (cambio solo de inferencia).
--preset elige el juego de configuraciones por defecto: lifecycle (las del ciclo de vida) o emb (las de emb_update sobre la mejor del
ciclo de vida: min_hits=3, motion=1, out_thresh=0.5).
Uso: python -m tracking.trackheads.track --ckpt <trackheads best.pth> --out <dir> [--configs ...]; luego tracking.baseline.evaluate
sobre <dir>/<nombre>/. O todo junto con slurm/trackheads_track.sbatch."""
import argparse, os, sys, time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tracking.baseline.mot_io import read_seqinfo, write_mot
from tracking.trackheads.data import seq_dirs
from tracking.trackheads.model import pair_iou
from tracking.trackheads.train import build_model
from tracking.video.frames import SequenceFrames

BIG = 1e9


def assign(score, null):
    """score (T,C) a maximizar, null (T,) puntuación de quedarse sin asignar -> idx (T,) long con la candidata o -1."""
    T, C = score.shape
    idx = torch.full((T,), -1, dtype=torch.long)
    if T == 0 or C == 0:
        return idx
    cost = np.full((T, C + T), BIG)
    cost[:, :C] = -score.double().cpu().numpy()
    cost[np.arange(T), C + np.arange(T)] = -null.double().cpu().numpy()
    r, c = linear_sum_assignment(cost)
    for i, j in zip(r, c):
        if j < C:
            idx[i] = int(j)
    return idx


CFG_KEYS = dict(min_hits=int, out_thresh=float, motion=int, motion_ema=float, motion_max=int, max_lost=int, null_bias=float, iou_min=float,
                birth_thresh=float, birth_nms=float, emb_update=str, emb_ema=float, clean_iou=float)
EMB_UPDATES = ('last', 'ema', 'clean', 'clean_ema')
_BEST = 'head,min_hits=3,motion=1,out_thresh=0.5'
PRESETS = {}
PRESETS['emb'] = [f'head_last={_BEST}', f'head_clean={_BEST},emb_update=clean', f'head_ema09={_BEST},emb_update=ema,emb_ema=0.9',
                  f'head_ema07={_BEST},emb_update=ema,emb_ema=0.7', f'head_clean_ema09={_BEST},emb_update=clean_ema,emb_ema=0.9',
                  f'head_clean_b06={_BEST},emb_update=clean,birth_thresh=0.6']
PRESETS['lifecycle'] = DEFAULT_CONFIGS = ['head=head', 'iou=iou', 'head_mh3=head,min_hits=3', 'head_mh3_mot=head,min_hits=3,motion=1',
                   'iou_mh3_mot=iou,min_hits=3,motion=1', 'head_mh3_mot_out05=head,min_hits=3,motion=1,out_thresh=0.5',
                   'head_mh3_mot_b06=head,min_hits=3,motion=1,birth_thresh=0.6']


def parse_config(spec, a):
    """"nombre=modo[,clave=valor...]" -> (nombre, Namespace con mode y todas las claves de CFG_KEYS; las no dadas, de la línea de comandos)."""
    name, rest = spec.split('=', 1)
    mode, *kv = rest.split(',')
    assert mode in ('head', 'iou'), f'modo desconocido en {spec!r}: {mode}'
    c = argparse.Namespace(mode=mode, **{k: getattr(a, k) for k in CFG_KEYS})
    for item in kv:
        k, v = item.split('=', 1)
        assert k in CFG_KEYS, f'clave desconocida en {spec!r}: {k} (válidas: {", ".join(CFG_KEYS)})'
        setattr(c, k, CFG_KEYS[k](v))
    assert c.emb_update in EMB_UPDATES, f'emb_update desconocido en {spec!r}: {c.emb_update} (válidos: {", ".join(EMB_UPDATES)})'
    return name, c


class Tracker:
    """Estado de un tracker: por track, id, emb y caja de su última detección asignada, frame de esa asignación, velocidad del centro,
    racha de asignaciones seguidas, nº de asignaciones y si está confirmado."""

    def __init__(self, cfg, model, dim=256, device='cpu'):
        self.mode, self.model, self.a, self.dev = cfg.mode, model, cfg, device
        z = lambda *s, **k: torch.zeros(*s, device=device, **k)
        self.ids, self.emb, self.box, self.last, self.vel = [], z(0, dim), z(0, 4), z(0), z(0, 2)
        self.streak, self.hits, self.conf = z(0, dtype=torch.long), z(0, dtype=torch.long), z(0, dtype=torch.bool)
        self.next_id = 1

    def ref_box(self, frame):
        """Caja de referencia de cada track en este frame: la última, con el centro extrapolado si motion."""
        if not self.a.motion or not len(self.ids):
            return self.box
        dt = (frame - self.last).clamp(max=self.a.motion_max)[:, None]
        c = (self.box[:, :2] + self.vel * dt).clamp(0, 1)
        return torch.cat([c, self.box[:, 2:]], -1)

    def scores(self, ref, emb_c, box_c, feats):
        """-> score (T,C) y null (T,) para los tracks vivos frente a las candidatas."""
        if self.mode == 'iou':
            return pair_iou(ref[None], box_c[None])[0], torch.full((len(self.ids),), self.a.iou_min, device=self.dev)
        T = len(self.ids)
        mask = torch.ones(1, T, dtype=torch.bool, device=self.dev)
        out = self.model(self.emb[None], ref[None], [f[None] for f in feats], mask)
        self.model.associate(out, emb_c[None], box_c[None], torch.ones(1, len(box_c), dtype=torch.bool, device=self.dev))
        logp = out['assoc_logits'][0].float().log_softmax(-1)
        return logp[:, :-1], logp[:, -1] + self.a.null_bias

    def update_emb(self, i, j, emb_c, box_c):
        a = self.a
        if a.emb_update in ('clean', 'clean_ema'):
            iou = pair_iou(box_c[j:j + 1][None], box_c[None])[0, 0]
            iou[j] = 0
            if len(iou) and iou.max() >= a.clean_iou:
                return                                      # caja solapada con otra candidata: se conserva la apariencia anterior
        if a.emb_update in ('ema', 'clean_ema'):
            self.emb[i] = a.emb_ema * self.emb[i] + (1 - a.emb_ema) * emb_c[j]
        else:
            self.emb[i] = emb_c[j]

    def _filter(self, keep):
        self.ids = [t for t, k in zip(self.ids, keep.tolist()) if k]
        for k in ('emb', 'box', 'last', 'vel', 'streak', 'hits', 'conf'):
            setattr(self, k, getattr(self, k)[keep])

    def step(self, frame, emb_c, box_c, sc_c, feats):
        """Un frame. emb_c (C,D), box_c (C,4), sc_c (C,) candidatas -> lista de (id, j) que se escriben en este frame
        (asignadas o nacidas, confirmadas y con score >= out_thresh)."""
        a, out = self.a, []
        early = frame <= a.min_hits                       # como OC-SORT: al principio de la secuencia no se espera a confirmar
        used = torch.zeros(len(sc_c), dtype=torch.bool, device=self.dev)
        if self.ids:
            score, null = self.scores(self.ref_box(frame), emb_c, box_c, feats)
            idx = assign(score, null)
            for i, j in enumerate(idx.tolist()):
                if j < 0:
                    continue
                used[j] = True
                dt = frame - self.last[i]
                inst = (box_c[j, :2] - self.box[i, :2]) / dt
                self.vel[i] = inst if self.hits[i] == 1 else a.motion_ema * self.vel[i] + (1 - a.motion_ema) * inst
                self.streak[i] = self.streak[i] + 1 if dt == 1 else 1
                self.hits[i] += 1
                self.conf[i] |= bool(self.streak[i] >= a.min_hits) or early
                self.update_emb(i, j, emb_c, box_c)
                self.box[i], self.last[i] = box_c[j], frame
                if self.conf[i] and sc_c[j] >= a.out_thresh:
                    out.append((self.ids[i], j))
            # muerte: más de max_lost frames sin asignar; un track sin confirmar muere en cuanto falla un frame
            self._filter(((frame - self.last) <= a.max_lost) & (self.conf | (self.last == frame)))
        born = []
        for j in torch.argsort(sc_c, descending=True).tolist():
            if used[j] or sc_c[j] < a.birth_thresh:
                continue
            ref = torch.cat([box_c[used], box_c[born]]) if born else box_c[used]
            if len(ref) and pair_iou(box_c[j:j + 1][None], ref[None]).max() >= a.birth_nms:
                continue
            born.append(j)
        if born:
            j = torch.tensor(born, device=self.dev)
            n = len(born)
            new = list(range(self.next_id, self.next_id + n))
            self.next_id += n
            conf = a.min_hits <= 1 or early
            self.ids += new
            self.emb, self.box = torch.cat([self.emb, emb_c[j]]), torch.cat([self.box, box_c[j]])
            self.last = torch.cat([self.last, torch.full((n,), float(frame), device=self.dev)])
            self.vel = torch.cat([self.vel, torch.zeros(n, 2, device=self.dev)])
            self.streak = torch.cat([self.streak, torch.ones(n, dtype=torch.long, device=self.dev)])
            self.hits = torch.cat([self.hits, torch.ones(n, dtype=torch.long, device=self.dev)])
            self.conf = torch.cat([self.conf, torch.full((n,), conf, dtype=torch.bool, device=self.dev)])
            if conf:
                out += [(t, k) for t, k in zip(new, born) if sc_c[k] >= a.out_thresh]
        return out


@torch.no_grad()
def track_sequence(det, model, seq_dir, a, device):
    """-> {modo: filas MOT (frame, id, x, y, w, h, score)} de una secuencia."""
    info = read_seqinfo(seq_dir)
    W, H = info['width'], info['height']
    trackers = {name: Tracker(cfg, model, device=device) for name, cfg in a.cfgs.items()}
    rows = {m: [] for m in trackers}
    dl = torch.utils.data.DataLoader(SequenceFrames(seq_dir), batch_size=a.batch, num_workers=a.num_workers, shuffle=False)
    for imgs, frames in dl:
        q, feats = det.infer(imgs.to(device))
        for b, frame in enumerate(frames.tolist()):
            cand = q['scores'][b] >= a.det_thr
            emb_c, box_c, sc_c = q['emb'][b][cand], q['boxes'][b][cand], q['scores'][b][cand]
            fb = [f[b] for f in feats]
            for m, tr in trackers.items():
                for tid, j in tr.step(frame, emb_c, box_c, sc_c, fb):
                    cx, cy, w, h = box_c[j].tolist()
                    rows[m].append((frame, tid, (cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H, float(sc_c[j])))
    return rows


def main(a, detector=None):
    device = torch.device(a.device if (a.device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    ck = torch.load(a.ckpt, map_location='cpu')
    ta = argparse.Namespace(**ck['args'])
    a.configs = a.configs or PRESETS[a.preset]
    a.cfgs = dict(parse_config(c, a) for c in a.configs)
    assert 'w_assoc' in ck['args'] or all(c.mode == 'iou' for c in a.cfgs.values()), \
        'el modo head necesita un checkpoint con Associator (trackheads_v3 o posterior)'
    model = build_model(ta).to(device).eval()
    model.load_state_dict(ck['model'])
    for k in ('det_ckpt', 'det_config', 'root'):
        if getattr(a, k) is None:
            setattr(a, k, getattr(ta, k))
    if a.det_thr is None:
        a.det_thr = ta.det_thr
    if detector is None:
        from tracking.trackheads.detector import FrozenDetector
        detector = FrozenDetector(a.det_ckpt, a.det_config, ta.top_k, device, not a.no_amp)
    seqs = seq_dirs(a.root, 'val', a.seqs, a.max_seqs)
    for sd in seqs:
        t0 = time.time()
        rows = track_sequence(detector, model, sd, a, device)
        s = os.path.basename(sd)
        for m, r in rows.items():
            write_mot(os.path.join(a.out, m, f'{s}.txt'), r)
        print(f'{s} ({time.time() - t0:.0f}s): ' + ', '.join(f'{m} {len(set(x[1] for x in r))} tracks {len(r)} cajas' for m, r in rows.items()), flush=True)
    return [os.path.basename(s) for s in seqs]


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='best.pth de TrackHeads (con Associator)')
    p.add_argument('--out', required=True, help='escribe <out>/<modo>/<seq>.txt')
    p.add_argument('--configs', nargs='+', default=None,
                   help='"nombre=modo[,clave=valor...]", modo head o iou; claves: ' + ', '.join(CFG_KEYS) + '. Por defecto, las de --preset')
    p.add_argument('--preset', choices=list(PRESETS), default='lifecycle', help='juego de configuraciones si no se da --configs')
    p.add_argument('--emb-update', choices=EMB_UPDATES, default='last', help='actualización del emb del track al asignarle una detección')
    p.add_argument('--emb-ema', type=float, default=0.9, help='[ema] peso del emb anterior')
    p.add_argument('--clean-iou', type=float, default=0.3, help='[clean] IoU con otra candidata a partir del cual la detección no es limpia')
    p.add_argument('--det-ckpt', default=None, help='por defecto, el del entrenamiento')
    p.add_argument('--det-config', default=None)
    p.add_argument('--root', default=None, help='DanceTrack (usa val/); por defecto, el del entrenamiento')
    p.add_argument('--seqs', nargs='*', default=None)
    p.add_argument('--max-seqs', type=int, default=0)
    p.add_argument('--det-thr', type=float, default=None, help='score mínimo de una candidata; por defecto, el del entrenamiento (0.3)')
    p.add_argument('--birth-thresh', type=float, default=0.5)
    p.add_argument('--birth-nms', type=float, default=0.7)
    p.add_argument('--max-lost', type=int, default=30, help='frames sin asignar antes de borrar un track')
    p.add_argument('--null-bias', type=float, default=0.0, help='[head] se suma a la log-prob de "ninguna" (>0: asigna menos)')
    p.add_argument('--min-hits', type=int, default=1, help='asignaciones seguidas para confirmar un track (1 = sin confirmación)')
    p.add_argument('--out-thresh', type=float, default=0.0, help='score mínimo de una caja para escribirla')
    p.add_argument('--motion', type=int, default=0, help='1 = extrapolar el centro con velocidad constante')
    p.add_argument('--motion-ema', type=float, default=0.5, help='media exponencial de la velocidad (peso de la anterior)')
    p.add_argument('--motion-max', type=int, default=10, help='frames máximos de extrapolación')
    p.add_argument('--iou-min', type=float, default=0.3, help='[iou] IoU mínimo para asignar')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--no-amp', action='store_true')
    return p


if __name__ == '__main__':
    main(parser().parse_args())
