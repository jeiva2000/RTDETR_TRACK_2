"""Configuración del módulo e2e: diccionario anidado con acceso por atributo, valores por defecto y sobrescritura `a.b=valor`.
Se carga desde tracking/configs/e2e.yml; `e2e.enabled: false` desactiva el módulo (solo detección)."""
import copy
import yaml

DEFAULTS = {
    'e2e': {'enabled': True},
    'tokens': {'top_k': 100, 'a_det': 'none'},                      # a_det: none | roialign | deform (hitos posteriores)
    'state': {'max_tracks': 64, 'max_age': 30},                      # max_age en frames del vídeo (no en pasos)
    'heads': {'dim': 256, 'hidden': 256, 'tau_init': 2.3},
    'appearance': {'mode': 'ema', 'alpha_app': 0.1},                 # H1: EMA fija (equivale a assoc 'ema' con alpha 0.9)
    'motion': {'mode': 'none', 'dim': 128, 'hidden': 128, 'vel_scale': 10.0},   # H2: mode none | gru (segundo vector por track con σ)
    'loss': {'w_assoc': 1.0, 'w_birth': 1.0, 'w_motion': 1.0, 'birth_pos_weight': 1.0, 'birth_detach': True},   # birth_detach: el gradiente del nacimiento no llega a la proyección compartida
    'train': {'T': 10, 'burn_in': 0, 'ss_max': 0.5, 'ss_warmup': 5, 'p_fn': 0.1, 'lr': 1e-3, 'weight_decay': 1e-4,
              'epochs': 20, 'batch': 8, 'samples_per_epoch': 2000, 'val_samples': 300, 'clip_grad': 1.0,
              'max_stride': 3, 'p_skip': 0.2, 'max_skip': 5},
    'infer': {'birth_mode': 'head', 'birth_thr': 0.5, 'birth_nms': 0.7, 'min_score': 0.0},   # birth_mode: head | score
    'h3': {                                                          # H3: asociación relacional (tracking/e2e/h3/); h3.enabled=true lo activa en lugar de E2EModel
        'enabled': False,
        'det_thr': 0.3,                                              # una query es detección si su score supera este valor (D variable)
        'K_hist': 10, 'app_dim': 128, 'dim': 256, 'heads': 8, 'ffn': 512, 'geo_hidden': 64,
        'layers_trk': 1, 'layers_det': 1, 'layers_x': 1,             # capas de auto-atención de tracks, de detecciones y de atención cruzada
        'gru_dim': 128, 'delta_scale': 10.0, 'sinkhorn_iters': 20, 'tau_init': 2.3,
        'branches': {'app': True, 'mot': True, 'rel': True},         # ablaciones: desactivar cada rama
        'gates': True, 'fixed_w': [1 / 3, 1 / 3, 1 / 3],             # gates por par; False = suma con pesos fijos (fixed_w)
        'memory': {'app_update': 'ema', 'alpha': 0.1, 'edges': False, 'edge_dim': 32},   # app_update: ema | gate; edges: memoria de aristas (fase 2)
        'loss': {'w_motion': 1.0, 'w_app': 1.0, 'tau_app': 0.1},     # L = L_assign + w_motion·L_motion + w_app·L_app
        'noise': {'jitter_pos': 0.05, 'jitter_scale': 0.05, 'p_fp': 0.03},   # el drop es train.p_fn
        'new_thr': 0.6,                                              # score mínimo de una detección sin asignar para crear un track (inferencia)
        'min_prob': 0.5,                                             # inferencia: una pareja del Hungarian se acepta si su probabilidad de Sinkhorn P_ij >= min_prob
        'birth_guard_iou': None, 'dedupe_iou': None,                 # inferencia, rompen el bucle de duplicados (h3/loop_breakers.py); None = apagado. dedupe_iou solo junto con birth_guard_iou
    },
    'ablation': {},                                                  # banderas de ablación de hitos posteriores
}


class Cfg(dict):
    """dict con acceso por atributo (anidado)."""
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def parse_overrides(items):
    """['train.lr=1e-3', 'appearance.mode=ema'] -> dict anidado (valores interpretados como YAML)."""
    out = {}
    for it in items or []:
        key, val = it.split('=', 1)
        node = out
        parts = key.split('.')
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return out


def load_config(path=None, overrides=None):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            deep_update(cfg, yaml.safe_load(f) or {})
    deep_update(cfg, parse_overrides(overrides) if isinstance(overrides, list) else (overrides or {}))
    return Cfg(cfg)
