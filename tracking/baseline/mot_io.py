"""Lectura/escritura de archivos en formato MOT: frame,id,x,y,w,h,conf,..."""
import configparser, os
import numpy as np


def read_seqinfo(seq_dir):
    c = configparser.ConfigParser(); c.read(os.path.join(seq_dir, 'seqinfo.ini'))
    s = c['Sequence']
    return dict(length=int(s['seqLength']), width=int(s['imWidth']), height=int(s['imHeight']),
                fps=float(s.get('frameRate', 20)), ext=s.get('imExt', '.jpg'))


def read_mot(path, full=False):
    """Devuelve array (N, 7): frame,id,x,y,w,h,conf (conf=1 si falta); con full=True conserva todas las columnas.
    Vacío si no hay filas."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return np.zeros((0, 7))
    a = np.loadtxt(path, delimiter=',', ndmin=2)
    if a.shape[1] < 7:
        a = np.concatenate([a, np.ones((len(a), 7 - a.shape[1]))], axis=1)
    return a if full else a[:, :7]


def by_frame(a):
    """array MOT -> {frame: filas}"""
    out = {}
    for f in np.unique(a[:, 0]).astype(int):
        out[f] = a[a[:, 0] == f]
    return out


def write_mot(path, rows):
    """rows: iterable (frame,id,x,y,w,h,score)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for fr, tid, x, y, w, h, s in rows:
            f.write(f'{int(fr)},{int(tid)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{s:.4f},-1,-1,-1\n')
