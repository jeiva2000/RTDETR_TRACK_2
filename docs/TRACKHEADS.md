# TrackHeads (`tracking/trackheads/`), etapa 1

Módulo de seguimiento sobre un RT-DETRv2 **congelado**. Cada detección del frame t recibe su propia cabeza de atención, generada por una
hipernetwork a partir de su query y su caja. Esa cabeza busca el mismo objeto en la memoria del encoder del frame t+k, predice su caja y si
sigue presente. No usa nada de `tracking/e2e/` salvo el helper `mlp`.

| Archivo | Contenido |
|---|---|
| `model.py` | `TrackHeads` (memoria, atención por cabeza, lectura, salida), `HeadGenerator` (hipernetwork: q_i, U_i, V_i), `sine_2d`, `token_centers`. |
| `detector.py` | `FrozenDetector`: `QueryExtractor` + hook en el encoder. `detect(images, gts)` -> top-K queries, track ID (Hungarian del detector) y los 3 mapas del encoder. |
| `data.py` | `PairDataset`: pares (t, t+k) de DanceTrack con k ~ U{k_min..k_max}; imágenes 640×640 y GT de los dos frames. Val: pares fijos. |
| `targets.py` | `build_tracks`: tracks del frame t y objetivos en t+k (presencia y caja GT). |
| `losses.py` | `trackheads_loss` (L1 + GIoU, BCE de presencia, atención auxiliar) y las métricas (`batch_stats`, `summarize`). |
| `train.py` | Bucle de entrenamiento, validación, checkpoints y reanudación. |
| `assoc.py` | Acierto de asociación: si la caja predicha (o la atención) apunta a la detección correcta de t+k, frente a hacerlo con prev_box; en total y en cruces. Es la métrica que elige `best.pth`. |
| `diagnose.py` | Diagnóstico de un checkpoint en val, por tramos de k y en cruces: acierto de asociación (pred, prev_box, atención); IoU de prev_box, de la predicción y techo (caja del detector en t+k); error de centro (prev, predicción, centro atencional) y de tamaño; masa de la atención en la caja GT y en prev_box; presencia. |
| `tests/test_model.py` | 12 tests en CPU (~5 s): formas, orden de los tokens, inicialización identidad, prior, independencia entre tracks, equivalencia de rango r, decodificación, gradientes y aprendizaje sintético. |
| `tests/test_assoc.py` | 3 tests con casos a mano de la métrica de asociación y de los cruces. |
| `tests/test_train.py` | 7 tests en CPU (~2 s) con un DanceTrack sintético en miniatura y un detector simulado: tracks y objetivos, enmascarado de la pérdida, muestreo de pares, humo de `main()` y reanudación. |
| `slurm/trackheads.sbatch` | Lanzamiento en el clúster (1 GPU). |

```python
from tracking.trackheads import TrackHeads
m = TrackHeads()          # D=256, d_h=64, r=8, 3 niveles, use_prior=True, log_scale=log(2), s_min=0.02
out = m(emb, prev_box, feats, track_mask)
# emb (B,T,256)  prev_box (B,T,4) cxcywh  feats [3×(B,256,H,W)] del frame t+k (strides 8/16/32)  track_mask (B,T)
# -> boxes (B,T,4), presence_logits (B,T), attn (B,T,N), token_xy (N,2), track_mask
```

## Decisiones de implementación (lo que la especificación no fijaba)
- **Orden de los tokens**: `feat.flatten(2)` fila a fila y niveles concatenados en orden 8, 16 y 32. El centro del token (i, j) es
  ((j+0.5)/W, (i+0.5)/H), el mismo orden que usa el encoder de RT-DETRv2.
- **seno2D**: senoidal tipo DETR sobre (x, y) en [0,1]·2π, con 128 dimensiones por eje y temperatura 10000. Le sigue `Linear(256,256)`.
  `level_embed` se inicializa con N(0, 0.02).
- **Claves y valores**: las claves salen de `x` (memoria con posición y nivel). La lectura x̄ = attn·`mem` usa la memoria **sin** posición,
  como dice la especificación.
- **Rango r sin materializar (B,T,N,d_h)**: como la corrección es lineal, (x̄ U_i) V_iᵀ tras promediar equivale a aplicar W_v + U_i V_iᵀ token
  a token. Hay un test que lo comprueba contra la versión explícita. El tensor más grande es attn (B,T,N): con 640×640, N = 8400.
- **Tracks de relleno**: se calculan igual (valores finitos y sin influencia sobre los demás tracks) y se devuelven tal cual. **Las pérdidas
  deben enmascararlos con `track_mask`.**
- Tamaño: unos 1.0 M de parámetros. Forward+backward con B=2, T=100, N=8400 tarda unos 0.05 s en CPU.

## Entrenamiento
```bash
python -m tracking.trackheads.tests.test_model && python -m tracking.trackheads.tests.test_train    # tests (CPU, segundos)
sbatch slurm/trackheads.sbatch      # entrena y al terminar diagnostica best.pth; detector: $OUTPUT_ROOT/detector_r50vd_person/best.pth (o DET_CKPT=...)
RUN_NAME=th_noattn EXTRA="--w-attn 0" sbatch slurm/trackheads.sbatch   # ablación: sin la pérdida auxiliar de atención
```
Salida en `$OUTPUT_ROOT/trackheads/<RUN_NAME>/` (por defecto `trackheads_v2`): `best.pth` (mejor métrica de `--select`), `last.pth` (reanudación), `log.txt` (una línea JSON por época), `args.json` y `diagnose.json` (el diagnóstico final, también impreso en el log del trabajo). Relanzar con el mismo `RUN_NAME` reanuda si se cortó por tiempo; si terminó, solo repite el diagnóstico. No reanuda una ejecución entrenada con otro `--select`.

**Receta.** Cada paso toma B pares (t, t+k) y hace **un** forward del detector congelado con los 2B frames. Los mapas del encoder no se
cachean (unos 8.6 MB por frame en fp32). Los **tracks** son las detecciones del frame t emparejadas con una identidad GT por el Hungarian
del detector, con IoU ≥ `--min-iou` (0.5) y score ≥ `--det-thr` (0.3); como mucho `--max-tracks` (64). `emb` y `prev_box` son los del
detector, no los del GT, como en inferencia. **Objetivos**: presente si la identidad está en el GT de t+k; en ese caso, la caja GT de t+k.
**Pérdida**: `5·L1 + 2·(1−GIoU)` solo en tracks presentes, más `1·BCE(presencia)` y `1·[−log masa de la atención dentro de la caja GT]`.
**Optimización**: AdamW con lr 1e-4 y wd 1e-4, warmup de 500 pasos y coseno, clip 1.0, 30 épocas de 8000 pares, parada tras 6 épocas sin
mejorar `val_iou`. k ~ U{1..10} (DanceTrack va a 20 fps, así que hasta 0.5 s).

**Qué se optimiza y qué se mide.** La pérdida es de caja y presencia, pero el objetivo es la **identidad**. La caja del detector en t+k
siempre será mejor que la predicha, así que la cabeza no tiene que competir con ella: tiene que apuntar a la detección correcta. Por eso
`best.pth` y la parada temprana usan `val_assoc_pred` (`--select assoc`, por defecto; `--select iou` recupera el criterio anterior). Esta
métrica es el acierto al elegir, entre las detecciones de t+k con score ≥ `--det-thr`, la de mayor IoU con la caja predicha. Se compara con
`val_assoc_prev` (lo mismo con prev_box: un tracker por IoU sin movimiento) y con `val_assoc_attn` (la detección con más masa de la
atención). Las mismas tres métricas en cruces (`val_cross_*`, cajas GT con IoU > `--cross-iou` 0.3 en t o t+k) son el caso donde prev_box
falla y la cabeza debería aportar. **No reanude con `--resume` una ejecución entrenada con el criterio anterior**: su mejor valor guardado es
un IoU y no se puede comparar con un acierto de asociación.

**Cómo leer el log**: `val_iou` frente a `val_iou_prev` (quedarse en la caja del frame t) y su diferencia `val_iou_gain`, que es lo que la
cabeza aporta. También `val_iou50` y `val_iou75` frente a sus `_prev`, `val_pres_acc` y `val_pres_recall_neg` (ausentes detectados),
`val_present_frac` y `val_attn_mass`. Los umbrales y métricas de val se fijan sobre val: el número es optimista, pero sirve para comparar ejecuciones.

**Sin probar con datos reales.** Los tests usan un detector simulado. `FrozenDetector` se comprobó con el RT-DETRv2 real del repo y pesos
aleatorios: 3 niveles 80²/40²/20², N = 8400, emparejamiento correcto y 0 parámetros entrenables. El tiempo por época y la memoria están sin medir.

**Límites conocidos de la etapa 1.**
- **Pocos negativos de presencia**: en DanceTrack casi nadie sale de escena en 0.5 s. Mire `val_present_frac`. Si la presencia no aprende,
  suba `--k-max` o baje `--pos-weight`. Todavía no hay negativos sintéticos (por ejemplo, la cabeza de un track contra la memoria de otra secuencia).
- Sin aumentos de datos y con una sola GPU (sin DDP).
- Los tracks vienen siempre de detecciones emparejadas con el GT: los falsos positivos del frame t no se usan como tracks.

## Observaciones de la etapa de diseño
- **Las dos inicializaciones a cero abren el gradiente en cascada**: en el paso 1 solo aprende la última capa de salida; en el paso 2, V y el
  resto; U recibe gradiente a partir del paso 3. Es el comportamiento esperado y no hace falta corregirlo.
- **Con la L1 de la caja como única pérdida, la atención aprende muy despacio.** En la tarea sintética de los tests (encontrar el patrón
  propio en la memoria, con la caja anterior desplazada ±2 tokens), 2000 pasos no bastan para mejorar a quedarse en `prev_box`. En cambio,
  supervisando la atención con −log attn en el token del objeto, la probabilidad sube de 0.007 a más de 0.6 en 600 pasos. Con datos reales
  puede ir mejor, porque las queries del decoder ya están alineadas con las features del encoder, pero conviene tener a mano una **pérdida
  auxiliar sobre la atención**: masa de la atención dentro de la caja GT del frame t+k, o distancia del centro atencional al centro GT.
