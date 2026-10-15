# Código de terceros (MIT)

Copiado de repos oficiales con cambios mínimos para ejecutarse sin `yolox`, `cython_bbox` ni `np.float`:

- `bytetrack/` ← https://github.com/ifzhang/ByteTrack @ d1bf019 (`yolox/tracker/`).
  Cambios: imports relativos, sin `torch`/`cv2`, `np.float`→`float`, `bbox_ious` (cython_bbox) reescrito en numpy
  con la misma convención de píxeles (+1), y `lap.lapjv(cost_limit)` emulado con `scipy.optimize.linear_sum_assignment`.
- `ocsort/` ← https://github.com/noahcao/OC_SORT @ 8462e7e (`trackers/ocsort_tracker/`). Sin cambios de código;
  requiere `filterpy` (y usa `lap` si está instalado, con alternativa en scipy).
- `trackeval/` ← https://github.com/JonathonLuiten/TrackEval @ 12c8791 (paquete `trackeval/`).
  Cambio: alias de NumPy retirados (`np.float`→`float`, `np.int`→`int`, `np.bool`→`bool`, `np.object`→`object`),
  porque el código oficial falla con NumPy >= 1.24. Las métricas no cambian. Se importa como `tracking.third_party.trackeval`.
