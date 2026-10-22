# Módulo de tracking e2e (`tracking/e2e/`)

Módulo de tracking construido **por hitos** sobre lo que ya existe en `tracking/` (extracción de queries, clips, caché, TrackEval, cruces, tablas con
semillas, SLURM). No modifica `rtdetrv2_pytorch/src/`. `tracking/assoc/` se conserva como referencia para comparar. Estado: **H1**, **H2** y **H3** implementados.

## Hito H1: tracker mínimo (detector congelado, caché)
Estado por track: apariencia `a` (nace de la query de su primera detección), caja y último frame visto. Afinidad track×detección = temperatura·coseno + rama
geométrica, con **columna nula** ("no aparece") por track y softmax por filas. Nacimiento: **cabeza binaria aparte** (BCE) por detección (D1; no se mezcla con
la softmax de los tracks). Actualización: EMA fija de la apariencia (`appearance.alpha_app`) solo con match. Todo vectorizado por lote con **relleno + máscaras**.
Misma receta de entrenamiento que `tracking/assoc/train.py` (clips con stride y saltos, scheduled sampling, FN inyectados) para que la comparación sea limpia.

| Archivo | Contenido |
|---|---|
| `config.py` | Valores por defecto, carga de `tracking/configs/e2e.yml` y sobrescritura `-u a.b=valor`. `e2e.enabled: false` desactiva el módulo. |
| `tokens.py` | `DetTokens` y el criterio único de top-K (`topk_queries`, el mismo de la caché y de inferencia). `a_det` llega en H5. |
| `state.py` | `TrackState` (B clips × N huecos): alive/ids/last, spawn, kill por `max_age`. Operaciones fuera de lugar (seguras con autograd). |
| `heads.py` | `AffinityHead` (coseno + geometría + columna nula) y `BirthHead`. Geometría y IoU vectorizados. |
| `losses.py` | Entropía cruzada de asociación (con columna nula, ignora relleno) y BCE de nacimiento. |
| `model.py` | `E2EModel`: `scores`, `decide` (Hungarian global o teacher forcing), `update` (EMA con/sin match). |
| `train.py` | Entrenamiento sobre la caché; guarda `best.pth` (mejor `val_acc` con decisiones propias) y `last.pth`. |
| `infer.py` | Inferencia frame a frame sobre la caché de val → MOT. Nacimientos: sin asignar, prob. de nacimiento y score mínimos, NMS. |
| `motion.py` | **H2**: `MotionHead` (GRU + predictor con σ), `motion_features`, `motion_loss`. Ver la sección del H2. |
| `tests/test_h1.py` | 8 tests (formas, relleno, actualización con/sin match, estado, asignación exclusiva, desactivación por config, humo). |

### Claves de configuración (`tracking/configs/e2e.yml`; defaults en `config.py`)
`e2e.enabled` · `tokens.{top_k,a_det}` · `state.{max_tracks,max_age}` · `heads.{dim,hidden,tau_init}` · `appearance.{mode,alpha_app}` ·
`loss.{w_assoc,w_birth,birth_pos_weight}` · `train.{T,burn_in,batch,epochs,samples_per_epoch,p_fn,ss_max,ss_warmup,lr,...}` · `infer.{birth_thr,birth_nms,min_score}`.
Las banderas de ablación de los hitos siguientes irán en `ablation`.

### Lanzar
```bash
python -m tracking.e2e.tests.test_h1                                     # tests (CPU, segundos)
# local o sin SLURM (una semilla: entrena, infiere y evalúa)
export FEAT_DIR=$OUTPUT_ROOT/features/detector_v1                         # caché train/ y val/ (tracking.video.cache_features)
bash tracking/scripts/e2e_one_seed.sh local e2e_h1 1
# servidor: 3 semillas en paralelo, y la tabla con las referencias de la Fase 6
BASE=e2e_h1 sbatch slurm/e2e_seeds.sbatch
bash tracking/scripts/aggregate_seeds.sh hpc e2e_h1 assoc_ema assoc_graph
```
`BIRTH_SCORE` (0.6) y `BIRTH_PROB` (0.5) fijan el score y la probabilidad mínimos de nacimiento en inferencia; **hay que barrerlos igual que el umbral de las variantes
`assoc` y de los baselines** antes de comparar en serio.

### Archivos añadidos para la tubería de servidor
| Archivo | Contenido |
|---|---|
| `tracking/e2e/train_joint.py` | Versión **descongelada**: detector (decoder + encoder + backbone, ver más abajo) + módulo e2e a la vez (BN en eval, `lr` del decoder 2e-5, detector congelado los primeros 1000 pasos si el módulo parte de cero, control de DetA con la **media de los umbrales 0.5/0.6/0.7**). **DDP** con `torchrun`: cada GPU procesa sus propios clips. |
| `tracking/e2e/dist_utils.py` | Inicialización de DDP (NCCL con GPU, gloo en CPU), difusión y suma entre procesos. Todo funciona igual sin entorno distribuido. |
| `tracking/e2e/sweep.py` | `run`: barrido de `birth_mode` / `birth_thr` / `min_score` sobre las semillas de una variante, evaluación final con la mejor. `select`: elige la mejor variante por HOTA medio. |
| `tracking/configs/e2e_variants.sh` | Lista de variantes (nombre\|overrides). Hoy solo claves del H1; los hitos siguientes añaden las suyas aquí. |
| `tracking/scripts/pipe_stage.sh` | Una etapa por invocación, **idempotente** (marcadores `.done`): `cache`, `baseline`, `variant`, `sweep`, `select`, `unfrozen`, `final`. |
| `tracking/scripts/pipe_job.sh` | Envoltorio de SLURM (módulos GCC/12.2.0 y CUDA/11.8.0, conda, etapa). |
| `tracking/scripts/e2e_pipeline.sh` | **Orquestador**: un solo comando para toda la tubería (SLURM con dependencias, o local repartiendo trabajos entre GPUs). |

## Hito H2: predictor de movimiento (segundo vector por track)
Con `motion.mode=gru` cada track lleva **dos vectores**: la apariencia `a` (EMA, como en H1) y el movimiento `m` (estado oculto de un GRU, `motion.dim`=128).
Con `motion.mode=none` (por defecto) el modelo es **exactamente el H1**: mismos parámetros, `state.m=None`, checkpoints del H1 compatibles.
- **Movimiento = delta de caja respecto a la última caja observada**, normalizado por su tamaño: `(dx/w, dy/h, log w'/w, log h'/h)`. Es la misma parametrización que la rama geométrica del H1.
- **Actualización (`MotionHead.observe`)**: el GRU recibe `[delta/dt, delta, dt/10]` **solo con match**; sin match `m` no cambia y el hueco (`gap`) crece. Un track nuevo parte con `m = 0`.
- **Predicción (`MotionHead.predict`)**: `(m, gap)` → media `mu` y `log σ` del delta para el frame actual (el `gap` entra como entrada: sirve con saltos y tras oclusiones). `log σ` en [-4, 1].
- **Uso en la afinidad**: a la rama geométrica se añaden, por par track×detección, `z = (observado - mu)/σ`, la NLL gaussiana y el IoU con la caja predicha (6 rasgos); la columna nula recibe además `m` y el `log σ` medio
  (un track con mucha incertidumbre tiene más probabilidad de "no aparecer"). La afinidad de apariencia no cambia.
- **Pérdida**: `loss.w_motion` × NLL gaussiana del delta de la detección con el **mismo ID**, sobre los tracks presentes y con el estado previo a la actualización (las cajas del objetivo están desconectadas del grafo).
  Se registra como `val_motion_nll`. Con DDP/descongelado funciona igual (cero conectado al grafo si no hay tracks presentes).
- **Claves**: `motion.{mode,dim,hidden,vel_scale}` y `loss.w_motion`. Variantes en `e2e_variants.sh`: `h2`, `h2_w01` (menos peso de la NLL), `h2_m64` (vector más pequeño).
```bash
python -m tracking.e2e.tests.test_h2                                                   # tests del H2 (CPU)
bash tracking/scripts/e2e_one_seed.sh local e2e_h2 1 -u motion.mode=gru                # una semilla: entrena, infiere y evalúa
python -m tracking.e2e.train --train-cache ... --val-cache ... --out ... -u motion.mode=gru
```
**Criterio de aceptación del H2**: mejorar al H1 (`h1`) más allá de la desviación entre semillas, sobre todo en cruces y tras oclusiones (AssA, IDF1, IDSW), con el mismo protocolo y el barrido de umbrales.
**Solo verificado con datos sintéticos en CPU** (el test de entrenamiento comprueba que el predictor aprende un movimiento de velocidad constante); la evaluación real está pendiente.
Limitaciones: la cámara se mueve poco en DanceTrack, así que el movimiento relativo a la caja es suficiente; no hay compensación de movimiento de cámara. La predicción es de un solo paso (con `gap` como entrada), sin multimodalidad.

## Hito H3: asociación relacional (`tracking/e2e/h3/`, `h3.enabled=true`)
Módulo alternativo que **sustituye** al `E2EModel` (H1/H2) con la misma tubería (clips, caché, entrenamiento congelado y descongelado con DDP, inferencia, barrido de umbrales, evaluación): `build_model(cfg)` elige el modelo.
Combina tres señales (apariencia, movimiento y **relaciones espaciales entre objetos**) con gates aprendidos por par detección–track. Con `h3.enabled=false` (por defecto) nada cambia para H1/H2.
Una detección es una query con score > `h3.det_thr` (D variable, con relleno y máscara); los tracks son los huecos vivos de la memoria (T variable). **Funciona con D = 0, T = 0 y T = 1** (por máscara y con tensores vacíos).

| Paso | Archivo (`tracking/e2e/h3/`) | Contenido |
|---|---|---|
| 1 | `appearance_head.py` | MLP(256→256→128) + L2, igual para detecciones y para la apariencia guardada de los tracks. |
| 2 | `motion_predictor.py` | GRU sobre las últimas K cajas (`[cx,cy,w,h,Δcx,Δcy,Δw,Δh]`, con padding y máscara): caja predicha en t, velocidad predicha y embedding de movimiento. |
| 3, 4, 5 | `relational_attention.py` | Auto-atención de tracks y de detecciones (pesos separados) con **sesgo aditivo en los logits** `B[h,i,j] = MLP_geo(g_ij)`; los tracks usan la caja **predicha** y la velocidad (6 términos), las detecciones, 4; atención cruzada detecciones←tracks (`s_rel` = logits de la última capa, media entre heads). |
| 6 | `affinity_gates.py` | `s_app` (coseno), `s_mot` (GIoU con la caja predicha), `s_rel`; gates por par con MLP(6→32→3) sobre `[s_app, s_mot, s_rel, margen_app, margen_mot, margen_rel]` (margen = top1 − top2 de la fila) y temperatura aprendible. |
| 7 | `assignment.py` | Matriz aumentada (D+1, T+1) con fila y columna **dustbin** (score `z` aprendible): Sinkhorn en log-dominio al entrenar (NLL contra la asignación GT); Hungarian sobre `s` al inferir y una pareja solo se acepta si su **probabilidad de Sinkhorn `P_ij` ≥ `h3.min_prob`** (0.5; la escala de `s` la fija el entrenamiento, `P` siempre está en [0, 1]) |
| 8 | `track_memory.py` | Memoria (apariencia, historial de K cajas, id, edad, frames desde la última asignación); actualización de la apariencia con **EMA o gate aprendido** (`h3.memory.app_update`), `max_age`, re-entrada. Memoria de **aristas** opcional (`h3.memory.edges`, fase 2): vector por pareja de tracks actualizado por GRU, sesgo extra en el paso 3. |
| — | `model.py`, `losses.py`, `noise.py`, `train.py`, `infer.py`, `geometry.py` | Ensamblado, pérdidas, ruido de entrenamiento, bucle de entrenamiento e inferencia. |

**Pérdidas** (`L = L_assign + w_motion·L_motion + w_app·L_app`, pesos `h3.loss.*`): `L_assign` NLL sobre la matriz de Sinkhorn contra la asignación GT, dustbin incluido (una pareja GT cuenta una vez; cada detección nueva o falso positivo y cada track no observado, una);
`L_motion` L1 + (1 − GIoU) entre la caja predicha y la GT del track; `L_app` InfoNCE (positivos = misma ID). Las IDs GT salen del emparejamiento de queries con el GT que ya usa la tubería (Hungarian del propio detector, descartando las parejas con IoU < 0.5), no de un Hungarian de IoU puro.
**Entrenamiento**: clips de L = `train.T` frames (la variante `h3` usa 8) con memoria; scheduled sampling como en H1/H2. **Fase 1** (detector congelado, caché): ruido sobre las cajas observadas (jitter de posición y escala, drop = `train.p_fn`, falsos positivos sintéticos `h3.noise.p_fp`).
**Fase 2** (descongelado): el mismo `train_joint.py` con DDP; el gradiente de `L_assign` y `L_app` llega al decoder por los embeddings (las cajas se desconectan).
**Bucle de duplicados y mecanismos de inferencia** (`h3/loop_breakers.py`, **apagados por defecto**): en val, el primer rechazo de una pareja crea un track duplicado, la probabilidad `P_ij` se reparte entre el original y la copia (~0.5/0.5, ninguna supera `h3.min_prob`),
se rechazan las dos y nacen más duplicados (en el informe: 13256 IDs frente a 273 identidades). En entrenamiento nunca nacen duplicados (solo nacen IDs GT sin track), así que el modelo no sabe salir de ahí; la validación del entrenamiento tampoco los ve. Dos banderas de inferencia:
`h3.birth_guard_iou` (una detección sin asignar no crea un track nuevo si un track vivo la cubre: IoU con su caja predicha ≥ umbral) y `h3.dedupe_iou` (se elimina el track más joven de dos tracks vivos solapados). **`dedupe_iou` sin la guarda empeora** (el duplicado eliminado renace al frame siguiente; sintético: IDs ×4 → ×11), por eso el barrido solo lo prueba con guarda.
Son un parche de inferencia, no la solución de fondo (entrenar con nacimientos decididos por el modelo).
**Ablaciones** (variantes en `e2e_variants.sh`): `h3_nogates` (pesos fijos), `h3_norel`, `h3_nomot`, `h3_noapp` (desactivan una rama), `h3_gate` (gate en la apariencia), `h3_edges` (memoria de aristas).
**Registro**: las partes de la pérdida salen por separado: `train_assign`, `train_app`, `train_motion` y `val_app_loss`, `val_motion_loss` (`val_loss` = `L_assign` media).
**Evaluación (etapa 3 del `.slurm`)**: no hay cabeza de nacimiento; el barrido recorre `h3.new_thr` (`--score`, 0.5 y 0.6) con `h3.min_prob` fijo en 0.5 (`--min-prob` admite varios valores para barrerlo), y `best_infer.json` los guarda como `min_score` (= `new_thr`) y `birth_thr` (= `min_prob`) con `birth_mode: h3`.
**Diferencias con la descripción original** (decisiones de integración): (i) las cajas que se perturban son las de las queries del detector (la caché no guarda el GT), y el objetivo de `L_motion` es la caja limpia de la query emparejada, no la caja GT original;
(ii) `box_pred = última caja + velocidad · frames desde la última asignación` (la velocidad queda supervisada por la pérdida de la caja); (iii) la memoria guarda la apariencia en el espacio del decoder (256) y se proyecta en cada frame;
(iv) los huecos de relleno se enmascaran con un valor finito (−1e4) en lugar de −inf, para que una fila sin llaves válidas no dé NaN.
**Solo verificado con datos sintéticos en CPU** (tests de formas y máscaras con D = 0, T = 0 y T = 1; invariancia al relleno; marginales de Sinkhorn; ablaciones; DDP con 2 procesos; el entrenamiento baja la pérdida de asignación). Falta medir tiempos y memoria en GPU y evaluar con DanceTrack real.
```bash
python -m tracking.e2e.tests.test_h3                                          # tests del H3 (CPU)
python -m tracking.e2e.train --train-cache ... --val-cache ... --out ... -u h3.enabled=true train.T=8      # fase 1 (caché)
E2E_RUN_VARIANTS="h3 h3_norel" sbatch --export=ALL slurm/e2e.slurm             # fase 2 y evaluación, variantes elegidas
```

### Diagnóstico del H3: dónde falla (`slurm/h3_diagnose.slurm`)
Trabajo corto (1 GPU, sin entrenar) que usa lo que dejó `e2e.slurm` en `$OUTPUT_ROOT/e2e_unfrozen_<variante>/` y contesta **dónde** falla la asociación, no solo cuánto. Lanzar: `sbatch slurm/h3_diagnose.slurm` (variables: `DIAG_VARIANT`, `DIAG_COMPARE`, `DIAG_MODES`, `DIAG_MAX_SEQS`, `DIAG_OVERRIDES`, `DIAG_RESWEEP`, `DIAG_MIN_PROB`, `DIAG_SCORES`).
1. **Resumen de los `log.txt`** de las variantes (`val_acc`, épocas, parada por DetA): ¿aprendió H3 a asociar tanto como H1/H2?
2. **`tracking.e2e.h3.diagnose`** sobre la caché de val, con el ID GT que guarda cada query (`track_id`), en dos modos: **`closed`** (el modelo y su memoria como en inferencia) y **`teacher`** (la memoria se actualiza con la asignación GT, así que cada decisión se evalúa con un estado limpio).
   Si en `teacher` el modelo acierta y en `closed` no, el problema es la acumulación de errores (desajuste entrenamiento–inferencia); si falla también en `teacher`, no ha aprendido a asociar. Mide, por modo:
   - el acierto de los tracks con su detección presente (correcta / rechazada sin asignar / enlazada a otra identidad / enlazada a un falso positivo) y de los ausentes (falso enlace);
   - el **top-1 de cada señal por separado** (apariencia, movimiento, relacional, afinidad con gates, probabilidad de Sinkhorn): si los gates empeoran la mejor señal suelta o una rama no discrimina, se ve aquí;
   - el peso medio de los gates, la distribución de `P_ij` de las parejas verdaderas y la **curva frente al umbral `h3.min_prob`** (parejas verdaderas que lo superan frente a falsos enlaces posibles);
   - la caja predicha por el GRU frente a la última caja observada;
   - en `closed`, las **causas de los cambios de ID** (`rebirth_prev_alive`: el track anterior seguía vivo y la detección nació como ID nuevo; `rebirth_after_kill`: murió por `max_age` antes de reaparecer; `swap_link`: enlace a otra identidad; `switch_back`), los nacimientos (identidad nueva, re-aparición, falso positivo, **perdidos por falta de hueco**) y los IDs creados frente a los del GT.
   El informe (`diagnose/report.md` y `.json`, y una copia en `$OUTPUT_ROOT/h3_diagnose_report.md`) acaba con **pistas heurísticas**: orientan, no demuestran.
   **Paso 2b** (`DIAG_LB_OVERRIDES`, activo por defecto con guarda 0.3 y dedupe 0.85): repite el diagnóstico en `closed` con los mecanismos contra duplicados y los compara con la ejecución original (IDs creados, cambios de ID, acierto, nacimientos perdidos por falta de hueco, tracks vivos por frame).
3. **Re-barrido con métricas reales** (HOTA, AssA, IDF1, IDSW) de `h3.min_prob` × `h3.new_thr` × guarda (`DIAG_GUARD`, por defecto 0, 0.3, 0.5) × dedupe (`DIAG_DEDUPE`, 0 y 0.85; solo con guarda) y, si se pide, `h3.det_thr` (`DIAG_DET_THR`), sobre una **copia** de la evaluación (no pisa `eval/seed_run`); la tabla sale ordenada por HOTA.
   Nota: hasta el commit `fe2047a` el barrido de `h3.min_prob` no llegaba al modelo (usaba una copia de la configuración): los re-barridos anteriores a ese arreglo no son válidos.
Tiempos sin medir; con `DIAG_MAX_SEQS=2 DIAG_RESWEEP=0` es una prueba rápida. La herramienta solo admite checkpoints de H3. Se puede lanzar a mano: `python -m tracking.e2e.h3.diagnose run --ckpt <e2e_best.pth> --cache <features/val> --out <dir> [-u h3.min_prob=0.3]` y `... logs --runs <e2e_unfrozen_h1> ...`.
Tests: `python -m tracking.e2e.tests.test_h3_diagnose` (los contadores suman el total; un umbral imposible se detecta y se atribuye a `rebirth_prev_alive`).

## Entrenamiento en el servidor con UN solo `.slurm`: detector + e2e descongelado
`slurm/e2e.slurm` sigue las convenciones de la plantilla del clúster (`--export=NONE`, `module purge`, comprobación de `nvcc`, `CUDA_HOME`, conda, diagnóstico en el log) y ejecuta,
en **un solo trabajo de un nodo y con DDP en ambas etapas** (una GPU por proceso):
1. **Entrenamiento de RT-DETR** (1 clase; train = DanceTrack + CrowdHuman, val = DanceTrack val; solver con early stopping sobre el AP) → `$OUTPUT_ROOT/$DET_RUN_NAME/best.pth`.
   **Cuándo se salta** (el `.slurm` lo decide al empezar y lo imprime con el prefijo `[detector]`): con `DET_CKPT_EXTERNAL=<ruta/best.pth>` (se usa ese checkpoint tal cual, de otra carpeta o ejecución); o si existe `$DET_RUN_NAME/best.pth` y hay marcador `TRAINING_DONE`;
   o si hay `best.pth` **sin `last.pth`** (un entrenamiento en curso siempre tiene `last.pth`, así que es un checkpoint final copiado o importado); o si hay `last.pth` pero el entrenamiento ya terminó (llegó a `DET_EPOCHS` o paró por early stopping, según `log.txt` y `early_stop.json`; se crea el marcador).
   Si el entrenamiento **no** terminó, **se reanuda** desde `last.pth`. Si hay marcador pero falta `best.pth`, se avisa y se reentrena. `FORCE_DET=1` reentrena siempre. Cuando se salta, la etapa 0 solo prepara DanceTrack (no descarga CrowdHuman ni rehace los COCO). El lote total es el menor múltiplo del nº de GPUs ≥ 16 (RT-DETR exige que sea divisible por las GPUs; el `lr` no se reescala).
2. **Cuando termina la 1**, entrenamiento **descongelado** del módulo e2e (detector + módulo, `tracking/e2e/train_joint.py`) partiendo de ese `best.pth`; cada GPU procesa sus propios clips.
   Se entrenan **de forma secuencial** las variantes de `E2E_RUN_VARIANTS` (por defecto **`h1 h2 h3`**: H1 = solo apariencia, H2 = + GRU de movimiento, H3 = asociación relacional), ambas desde el mismo detector, cada una en
   `$OUTPUT_ROOT/e2e_unfrozen_<variante>/` (`E2E_RUN_PREFIX` cambia el prefijo). Todas las variantes se validan antes de empezar. Cada una deja un marcador `TRAINING_DONE` que guarda la **receta** con la que se entrenó (segunda línea: `E2E_TRAIN_REV`, overrides de la variante, `LAMBDA`, `CLIPS_PER_GPU` y `JOINT_ARGS`; no incluye el nº de GPUs).
   Al relanzar se **salta** la variante cuya receta coincide con la actual; si **no coincide** (o el marcador es antiguo y no tiene firma, que cuenta como receta 1: solo decoder) se **reentrena sola**, y `FORCE_E2E=1` las repite todas.
   Una ejecución previa (a medias o con otra receta) se aparta como `<carpeta>_previo_<jobid>` (no se borra; el entrenamiento e2e no tiene reanudación), así que **cada reentrenamiento deja una copia más en disco** (checkpoints del detector incluidos): bórralas a mano cuando no hagan falta.
   `E2E_TRAIN_REV` (2 = detector completo: backbone + encoder + decoder; 1 = solo decoder) se sube cuando cambia la receta y fuerza el reentrenamiento. Si una variante **falla**, las demás se ejecutan igualmente y el trabajo
   termina con error al final (el resumen lista entrenadas / saltadas / fallidas). Elegir otras: `sbatch --export=ALL,E2E_RUN_VARIANTS="h2 h2_w01" slurm/e2e.slurm`; una sola: `E2E_RUN_VARIANTS=h2` (o `E2E_VARIANT=h2`).
   El tiempo es la suma de las variantes: ajusta `-t` de la cabecera `#SBATCH` (aún sin medir).
3. **Evaluación de cada variante** (etapa 3; `RUN_EVAL=0` la desactiva). Para cada variante entrenada, en `$OUTPUT_ROOT/e2e_unfrozen_<variante>/eval/`:
   - caché de val con **su** detector ajustado (`detector_best.pth`);
   - **ByteTrack y OC-SORT con ese mismo detector**, a 0.6 y barriendo `track_thresh` (reutiliza la etapa `baseline` de `pipe_stage.sh`);
   - barrido de los umbrales de nacimiento del módulo (`birth_mode`, `birth_thr`, `min_score`) con `e2e_best.pth` y evaluación completa con el mejor: **HOTA, DetA, AssA, IDF1, IDSW, global y en cruces**, con TrackEval (`eval/seed_run/eval.json`; la configuración elegida, en `best_infer.json`).

   Al final se escribe la tabla comparativa de todas las variantes en **`$OUTPUT_ROOT/e2e_unfrozen_comparacion.md`** (filas `<variante>_e2e`, `<variante>_bytetrack_0.6`, `<variante>_bytetrack_<mejor umbral>`, igual para OC-SORT) y la configuración de inferencia elegida.
   Cada variante deja el marcador `eval/EVAL_DONE` y se salta al relanzar (`FORCE_EVAL=1` repite la evaluación, baselines incluidos); un fallo en una variante no impide evaluar las demás, y el trabajo sale con error al final si algo falló.
   **Cómo leerla:** los umbrales se eligen sobre val y se reportan sobre val (optimista; la misma regla para todos los métodos). Es **una semilla por ejecución**: diferencias de pocos puntos de HOTA entre H1 y H2 no son concluyentes sin repetir con otra `SEED`
   (±2–3 puntos de HOTA entre semillas en sintético, sin medir con datos reales). Cada variante tiene su propio detector ajustado, así que sus baselines no son idénticos entre variantes.
   Tiempo extra: caché de val + detección de val + barrido de umbrales por variante (sin medir); ajusta `-t`.

Por ahora **no** se ejecutan las variantes con detector congelado (siguen disponibles en `tracking/scripts/e2e_pipeline.sh`).
```bash
cd ~/work/RTDETR_TRACK_2 && git pull                            # logs/ ya viene en el repo (Slurm la necesita antes de arrancar)
cp setup_env/paths.hpc.env.example setup_env/paths.hpc.env      # rutas absolutas (si aún no existe); las salidas van por defecto a <repo>/outputs
# edita en slurm/e2e.slurm: la cabecera #SBATCH (--partition, --gres, --nodelist, --cpus-per-task, --mem, -t) y el bloque CONFIGURACIÓN
sbatch slurm/e2e.slurm
```
**Salidas:** por defecto en `outputs/` dentro del repositorio (`OUTPUT_ROOT`, ignorada por git; fíjala en `paths.hpc.env` si prefieres otra ruta, p. ej. otro disco: los checkpoints y las cachés ocupan varios GB).

**Datos que deben estar preparados antes (Fase 2):** el COCO de train (`$COCO_DIR/detector_train/annotations/train.json`, mezcla de DanceTrack y CrowdHuman con `merge_coco.py`), el COCO de val
(`$COCO_DIR/dancetrack/annotations/val.json`), DanceTrack en `$DANCETRACK_DIR` (`train1/ train2/ val/`) y el checkpoint COCO en `$CKPT_DIR`. El script lo comprueba al empezar y, si falta el checkpoint COCO,
intenta descargarlo (los nodos de cómputo pueden no tener internet: descárgalo antes desde el login con `bash setup_env/download_ckpt.sh hpc`). Si el trabajo se corta en la etapa 2, al relanzarlo la etapa 1 se salta
pero la 2 empieza de nuevo (el entrenamiento e2e no tiene reanudación). Con `sbatch --export=ALL,FORCE_DET=1 slurm/e2e.slurm` se reentrena el detector aunque exista.

**Entorno conda:** si el entorno de `CONDA_ENV` no existe, el propio `.slurm` lo **crea** (Python 3.10 con `conda-forge`, torch 2.1.2 + torchvision 0.16.2 con el índice `cu118`, los requisitos de RT-DETRv2 y de `tracking/`, y `numpy<2`) y lo verifica; si ya existe, **solo lo activa**. Detalles:
un candado (`flock`) evita que dos trabajos lo creen a la vez; si la creación falla o se cancela, el entorno a medias se borra; el nodo necesita internet para crearlo (se comprueba y, si no hay, el trabajo avisa de que lo crees desde el login con
`bash setup_env/setup_hpc.sh`); `CREATE_ENV=0` desactiva la creación; `CONDA_ENV` puede ser un nombre o una ruta absoluta. El índice de torch (`TORCH_INDEX`) debe corresponder a la CUDA del módulo (11.8 → `cu118`).
La creación suma varios minutos de descargas al tiempo del trabajo.

## Tubería completa con un trabajo por etapa (alternativa, desde el login)
```bash
cd ~/work/RTDETR_TRACK_2 && git pull && mkdir -p logs
export DET_CKPT=/ruta/detector/best.pth FEAT_DIR=/ruta/features/detector_v1
export SBATCH_COMMON="--partition=<X> --account=<Y>"       # y, si hace falta, SBATCH_GPU="--gres=gpu:a100:1"
DRY_RUN=1 bash tracking/scripts/e2e_pipeline.sh            # revisar primero qué enviaría
bash tracking/scripts/e2e_pipeline.sh                      # con SLURM: envía todo con dependencias y termina; sin SLURM usa el modo local
```
Orden y dependencias: `cache` ∥ `baseline` → variantes congeladas (NSEEDS semillas, 5 por defecto, tras la caché) → `sweep` por variante → `select` →
`unfrozen` (un trabajo por λ, por defecto 0, 3 y 10; el 0 es el control) → `final`. Resultados en `$OUTPUT_ROOT/e2e_pipe/` (cambia con `PIPE_NAME`); la tabla en `final_table.md`.
Es **reanudable**: lo ya hecho deja un marcador y se salta (`FORCE=1` lo repite). Sin SLURM: `MODE=local NGPU=5 bash tracking/scripts/e2e_pipeline.sh` (en `tmux` o `nohup`).
Variables útiles: `NSEEDS`, `LAMBDAS`, `VARIANTS="h1 h1_a02"`, `E2E_EXTRA_OV="train.epochs=30"`, `JOINT_ARGS="--epochs 40 --clips-per-step 4"`, `SWEEP_ARGS`, `TIME_*`.

### Qué se entrena del detector en el ajuste conjunto (`train_joint.py`)
**Por defecto, el detector completo**, como en el entrenamiento por defecto de RT-DETRv2 (config `rtdetrv2_r50vd_6x_coco.yml`): decoder, encoder y backbone, **salvo la primera capa del backbone** (`freeze_at: 0`) **y las normalizaciones del backbone**
(`freeze_norm: True`, `FrozenBatchNorm`), que siguen congeladas. Todo el detector va en **eval** (BatchNorm con estadísticas fijas: con lotes de frames de un mismo clip, muy correlacionados, las del lote no son las de la población; el dropout del encoder es 0).
`--no-train-backbone` y `--no-train-encoder` los dejan congelados y sin gradiente (con el backbone entrenable y el encoder congelado, el gradiente atraviesa el encoder igualmente).
| Parte | Entrenable por defecto | `lr` por defecto | Opción |
|---|---|---|---|
| Decoder | sí | `--lr-det` 2e-5 | — |
| Encoder | sí | `--lr-encoder` (= `--lr-det`) | `--no-train-encoder` |
| Backbone (sin 1.ª capa ni normalizaciones) | sí | `--lr-backbone` (= `--lr-det` / 10) | `--no-train-backbone` |
Las proporciones entre `lr` (backbone 10 veces menor que encoder y decoder) son las de RT-DETRv2 (1e-5 frente a 1e-4); los valores absolutos y el desfase de 5 veces respecto a RT-DETRv2 **no están validados**. Hay un solo recorte de gradiente (`--clip-det`) para todo el detector.
Con el módulo desde cero, los primeros `--det-freeze-steps` pasos todo el detector va a `lr` 0 (solo aprende el módulo). El `detector_best/last.pth` guarda el modelo completo (backbone y encoder incluidos), que es el que usan después la caché de val y los baselines de la etapa 3.
**Coste y riesgo:** entrenar el backbone cuesta bastante más memoria y tiempo por paso (T frames de 640×640 con gradiente) y no está medido en GPU; con el control de DetA (tolerancia 0.5) el ajuste puede pararse antes. DDP sincroniza los gradientes del backbone y del encoder (se registran en `JointModule`) y `--check-sync` comprueba el detector completo.
Ejemplos: `JOINT_ARGS="--epochs 40 --samples-per-epoch 1000 --no-train-backbone"` (solo encoder + decoder) · `--no-train-backbone --no-train-encoder` (como antes: solo decoder).

### Varias GPUs en un entrenamiento (DDP)
`tracking/e2e/train_joint.py` se lanza con `torchrun`; **cada proceso (una GPU) procesa `--clips-per-gpu` clips distintos por paso** (`DistributedSampler`), y los gradientes se
sincronizan con DDP. El lote efectivo es clips-por-GPU × nº de GPUs, y `--samples-per-epoch` cuenta los clips de **todas** las GPUs. Decisiones de diseño:
- **Un único módulo** (`JointModule`: decoder, encoder y backbone si son entrenables + módulo e2e) cuyo `forward` devuelve la pérdida total, para que DDP sincronice todos los gradientes con un solo `backward`.
- **`find_unused_parameters=True`:** en un paso algún parámetro del módulo puede no recibir gradiente (p. ej. un clip sin personas); se mantiene además la pérdida "cero" conectada al grafo. `broadcast_buffers=False`: las BatchNorm del decoder están en eval.
- **Evaluación repartida:** el DetA (secuencias `rank::world`) y la precisión en vivo (clips `rank::world`) se calculan en paralelo y se combinan; TrackEval lo ejecuta solo el proceso principal. Solo este escribe log y checkpoints.
- **Aleatoriedad:** misma semilla para construir (y DDP difunde desde el rank 0), semilla distinta por proceso para FN inyectados y scheduled sampling.
- **`lr`:** sin reescalado por defecto; `--lr-scale linear|sqrt` es opcional y está sin validar. Los pasos de congelado del decoder y de calentamiento cuentan pasos globales.
```bash
# 5 GPUs en un nodo, un clip por GPU (lote efectivo 5)
torchrun --standalone --nproc_per_node=5 -m tracking.e2e.train_joint --det-ckpt $DET_CKPT --out $OUTPUT_ROOT/unfrozen \
  --clips-per-gpu 1 --samples-per-epoch 1000 --epochs 40 --lambda-e2e 10
# dentro de la tubería: UNFROZEN_GPUS=5 (y CLIPS_PER_GPU, por defecto 1 con varias GPUs)
UNFROZEN_GPUS=5 LAMBDAS="0 10" bash tracking/scripts/e2e_pipeline.sh     # SLURM: pide gpu:5 por trabajo (SBATCH_GPU_TEMPLATE="--gres=gpu:a100:%N")
```
En modo local los trabajos se lanzan en oleadas de `NGPU / UNFROZEN_GPUS` a la vez, cada uno con sus GPUs. Con `UNFROZEN_GPUS=5` los valores de λ se entrenan uno tras otro; con 1, en paralelo.
Test: `python -m tracking.e2e.tests.test_ddp` (2 procesos en CPU, con H1 y con H2; comprueba que las réplicas siguen idénticas). Si `torchrun --standalone` se cuelga por un nombre de equipo que no resuelve, el test ya fija `--local-addr 127.0.0.1`.

### Qué contiene la tabla final
Baselines (ByteTrack y OC-SORT a 0.6 **y a su mejor umbral del barrido**), cada variante congelada **a su mejor umbral de inferencia** (media ± desviación entre semillas, global y cruces)
y cada ejecución descongelada (un valor de λ por fila). Los umbrales se eligen sobre val y se reportan sobre val: es optimista, pero se aplica la misma regla a todos.

### Límites conocidos (léelos antes de lanzar)
- **Solo están H1, H2 y H3** (q_det + caja; EMA fija, GRU de movimiento opcional y la asociación relacional del H3). Banco histórico, `a_det` y ruido (H4–H6) **no están implementados**; "la versión descongelada" es el módulo (H1 o H2 según la variante elegida) con el decoder descongelado.
- **DDP probado solo con 2 procesos en CPU (`gloo`)**, no con GPUs ni NCCL; los tiempos y la memoria por GPU están sin medir. Se asume un **sistema de archivos compartido** entre procesos (la evaluación escribe archivos que lee el proceso principal).
- **Sin medir en el clúster:** tiempos, memoria y `--time` de los trabajos son estimaciones; los `--gres`, `--partition` y `--account` se pasan por `SBATCH_*`.
- La tubería se probó **en local con datos sintéticos** (etapas `variant`, `sweep`, `select`, `final`); `cache`, `baseline` y `unfrozen` se probaron por separado y con datos sintéticos o en `DRY_RUN`, no encadenadas con datos reales.
- Con 5 semillas la variación del HOTA entre ejecuciones es de ±2–3 puntos (sintético); diferencias menores no son concluyentes.

### Criterio de aceptación del H1
Igualar a `ema` (misma actualización) y no quedar por debajo de `graph` más allá de la desviación entre semillas, con el mismo protocolo (3 semillas, HOTA/DetA/AssA/IDF1/IDSW
globales y en cruces). **Solo se ha verificado con datos sintéticos en CPU**; la comprobación real es en la workstation o el clúster.

### Próximos hitos
H3 asociación relacional (ya implementado) · H4 banco histórico · H5 `a_det` (RoIAlign; vector por query en la caché mientras el detector esté
congelado) · H6 ruido de entrenamiento · H7 descongelar el decoder (BN en eval, DDP, control de AP y DetA; ya implementado). H2 (predictor de movimiento) ya implementado.

## Preparación automática de datos (etapa 0 de `slurm/e2e.slurm`)

Antes de entrenar, el `.slurm` comprueba y prepara solo lo que falta (idempotente, con `flock` en `$DATA_ROOT/.data_prep.lock`):

1. DanceTrack en `$DANCETRACK_DIR` (20/20/25 secuencias con `gt.txt`) → `setup_env/download_dancetrack.sh`.
2. COCO de DanceTrack en `$COCO_DIR/dancetrack/annotations/{train,val}.json` → `tracking/data/convert_dancetrack.py`.
3. Si `USE_CROWDHUMAN=1` (por defecto): CrowdHuman (`download_crowdhuman.sh … full`) y su COCO (`convert_crowdhuman.py`).
4. Mezcla `$COCO_DIR/detector_train/annotations/train.json` → `merge_coco.py` (se rehace si una entrada es más nueva).

Variables: `PREPARE_DATA` (0 = solo comprobar), `USE_CROWDHUMAN` (0 = detector solo con DanceTrack; se aparta del plan), `DT_STRIDE_TRAIN`, `DT_STRIDE_VAL`.
Descargar necesita internet en el nodo de cómputo; si no lo hay el trabajo falla con un mensaje que indica el comando a ejecutar antes desde el login.
Un JSON cortado (escritura interrumpida) se detecta y se regenera; una descarga/extracción interrumpida de DanceTrack o CrowdHuman hay que borrarla a mano.
Verificado solo con stubs (sin descargas reales): primera pasada ejecuta todo en orden; segunda no ejecuta nada.
