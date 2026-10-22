"""Rompe el bucle de duplicados en inferencia. Observado en val (informe del diagnóstico): un rechazo crea un track duplicado, la probabilidad P_ij se reparte entre el original y la
copia (~0.5/0.5, ninguna supera h3.min_prob), se rechazan las dos, nacen más duplicados... En entrenamiento nunca nacen duplicados (solo nacen IDs GT sin track), así que el modelo no sabe
salir de ahí. Dos mecanismos de inferencia, apagados por defecto (None / 0) y recorridos por el barrido (sweep.py):
- birth_guard (h3.birth_guard_iou): una detección sin asignar NO crea un track nuevo si un track vivo ya la cubre (IoU de su caja con la caja predicha del track >= umbral).
- duplicate_tracks (h3.dedupe_iou): si dos tracks vivos se solapan (IoU de sus últimas cajas >= umbral), se elimina el MÁS JOVEN (nació después; empate: id mayor).
ATENCIÓN: la eliminación de duplicados SOLA empeora (el duplicado eliminado deja su detección sin asignar y nace otra vez en el frame siguiente: en el caso sintético forzado los IDs pasan
de x4 a x11); úsala siempre junto con la guarda, que impide ese renacimiento (el barrido solo prueba dedupe con guarda).
Los usan por igual tracking/e2e/h3/infer.py y el diagnóstico (diagnose.py)."""
import torch

from tracking.e2e.heads import box_iou_b


def birth_guard(det_box, trk_box, alive, thr):
    """det_box (K,4), trk_box (N,4) caja de referencia de cada track, alive (N,) -> (K,) bool: True si la detección ya está cubierta por un track vivo. Sin umbral: todo False."""
    K = det_box.shape[0]
    if not thr or K == 0 or not bool(alive.any()):
        return torch.zeros(K, dtype=torch.bool, device=det_box.device)
    iou = box_iou_b(det_box[None], trk_box[None])[0]                         # (K,N)
    return iou.masked_fill(~alive[None, :], 0.0).max(1).values >= thr


def duplicate_tracks(boxes, alive, born, ids, thr):
    """boxes (N,4) última caja de cada track, alive/born/ids (N,) -> (N,) bool: True para el track más joven de cada pareja de tracks vivos con IoU >= thr. Sin umbral: todo False."""
    N = boxes.shape[0]
    if not thr or int(alive.sum()) < 2:
        return torch.zeros(N, dtype=torch.bool, device=boxes.device)
    iou = box_iou_b(boxes[None], boxes[None])[0]                             # (N,N)
    pair = (iou >= thr) & alive[:, None] & alive[None, :] & ~torch.eye(N, dtype=torch.bool, device=boxes.device)
    younger = (born[:, None] > born[None, :]) | ((born[:, None] == born[None, :]) & (ids[:, None] > ids[None, :]))      # i es más joven que j
    return (pair & younger).any(1)
