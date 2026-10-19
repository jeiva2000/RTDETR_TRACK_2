# Módulo de tracking e2e (`tracking/e2e/`)

Módulo de tracking construido **por hitos** sobre lo que ya existe en `tracking/` (extracción de queries, clips, caché, TrackEval, cruces, tablas con
semillas, SLURM). No modifica `rtdetrv2_pytorch/src/`. `tracking/assoc/` se conserva como referencia para comparar. Estado: **H1** implementado.

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
| `tracking/e2e/train_joint.py` | Versión **descongelada**: decoder + módulo e2e a la vez (BN del decoder en eval, `lr` del decoder 2e-5, decoder congelado los primeros 1000 pasos si el módulo parte de cero, control de DetA con la **media de los umbrales 0.5/0.6/0.7**). **DDP** con `torchrun`: cada GPU procesa sus propios clips. |
| `tracking/e2e/dist_utils.py` | Inicialización de DDP (NCCL con GPU, gloo en CPU), difusión y suma entre procesos. Todo funciona igual sin entorno distribuido. |
| `tracking/e2e/sweep.py` | `run`: barrido de `birth_mode` / `birth_thr` / `min_score` sobre las semillas de una variante, evaluación final con la mejor. `select`: elige la mejor variante por HOTA medio. |
| `tracking/configs/e2e_variants.sh` | Lista de variantes (nombre\|overrides). Hoy solo claves del H1; los hitos siguientes añaden las suyas aquí. |
| `tracking/scripts/pipe_stage.sh` | Una etapa por invocación, **idempotente** (marcadores `.done`): `cache`, `baseline`, `variant`, `sweep`, `select`, `unfrozen`, `final`. |
| `tracking/scripts/pipe_job.sh` | Envoltorio de SLURM (módulos GCC/12.2.0 y CUDA/11.8.0, conda, etapa). |
| `tracking/scripts/e2e_pipeline.sh` | **Orquestador**: un solo comando para toda la tubería (SLURM con dependencias, o local repartiendo trabajos entre GPUs). |

## Entrenamiento en el servidor con UN solo `.slurm`: detector + e2e descongelado
`slurm/e2e.slurm` sigue las convenciones de la plantilla del clúster (`--export=NONE`, `module purge`, comprobación de `nvcc`, `CUDA_HOME`, conda, diagnóstico en el log) y ejecuta,
en **un solo trabajo de un nodo y con DDP en ambas etapas** (una GPU por proceso):
1. **Entrenamiento de RT-DETR** (1 clase; train = DanceTrack + CrowdHuman, val = DanceTrack val; solver con early stopping sobre el AP) → `$OUTPUT_ROOT/$DET_RUN_NAME/best.pth`.
   Si ya está hecho (marcador `TRAINING_DONE`) se salta; si se cortó por tiempo, **se reanuda** desde `last.pth`. El lote total es el menor múltiplo del nº de GPUs ≥ 16 (RT-DETR exige que sea divisible por las GPUs; el `lr` no se reescala).
2. **Cuando termina la 1**, entrenamiento **descongelado** del módulo e2e (decoder + módulo, `tracking/e2e/train_joint.py`) partiendo de ese `best.pth`; cada GPU procesa sus propios clips → `$OUTPUT_ROOT/$E2E_RUN_NAME/`.

Por ahora **no** se ejecutan las variantes con detector congelado, el barrido de umbrales ni las evaluaciones (siguen disponibles en `tracking/scripts/e2e_pipeline.sh`).
```bash
cd ~/work/RTDETR_TRACK_2 && git pull                            # logs/ ya viene en el repo (Slurm la necesita antes de arrancar)
cp setup_env/paths.hpc.env.example setup_env/paths.hpc.env      # rutas absolutas (si aún no existe)
# edita en slurm/e2e.slurm: la cabecera #SBATCH (--partition, --gres, --nodelist, --cpus-per-task, --mem, -t) y el bloque CONFIGURACIÓN
sbatch slurm/e2e.slurm
```
**Datos que deben estar preparados antes (Fase 2):** el COCO de train (`$COCO_DIR/detector_train/annotations/train.json`, mezcla de DanceTrack y CrowdHuman con `merge_coco.py`), el COCO de val
(`$COCO_DIR/dancetrack/annotations/val.json`), DanceTrack en `$DANCETRACK_DIR` (`train1/ train2/ val/`) y el checkpoint COCO en `$CKPT_DIR`. El script lo comprueba al empezar y, si falta el checkpoint COCO,
intenta descargarlo (los nodos de cómputo pueden no tener internet: descárgalo antes desde el login con `bash setup_env/download_ckpt.sh hpc`). Si el trabajo se corta en la etapa 2, al relanzarlo la etapa 1 se salta
pero la 2 empieza de nuevo (el entrenamiento e2e no tiene reanudación). Con `sbatch --export=ALL,FORCE_DET=1 slurm/e2e.slurm` se reentrena el detector aunque exista.

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

### Varias GPUs en un entrenamiento (DDP)
`tracking/e2e/train_joint.py` se lanza con `torchrun`; **cada proceso (una GPU) procesa `--clips-per-gpu` clips distintos por paso** (`DistributedSampler`), y los gradientes se
sincronizan con DDP. El lote efectivo es clips-por-GPU × nº de GPUs, y `--samples-per-epoch` cuenta los clips de **todas** las GPUs. Decisiones de diseño:
- **Un único módulo** (`JointModule`: decoder + módulo e2e) cuyo `forward` devuelve la pérdida total, para que DDP sincronice todos los gradientes con un solo `backward`.
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
Test: `python -m tracking.e2e.tests.test_ddp` (2 procesos en CPU, comprueba que las réplicas siguen idénticas).

### Qué contiene la tabla final
Baselines (ByteTrack y OC-SORT a 0.6 **y a su mejor umbral del barrido**), cada variante congelada **a su mejor umbral de inferencia** (media ± desviación entre semillas, global y cruces)
y cada ejecución descongelada (un valor de λ por fila). Los umbrales se eligen sobre val y se reportan sobre val: es optimista, pero se aplica la misma regla a todos.

### Límites conocidos (léelos antes de lanzar)
- **Solo está el H1** (q_det + caja, EMA fija). Movimiento, grafo, banco, `a_det` y ruido (H2–H6) **no están implementados**; "la versión descongelada" es el H1 con el decoder descongelado.
- **DDP probado solo con 2 procesos en CPU (`gloo`)**, no con GPUs ni NCCL; los tiempos y la memoria por GPU están sin medir. Se asume un **sistema de archivos compartido** entre procesos (la evaluación escribe archivos que lee el proceso principal).
- **Sin medir en el clúster:** tiempos, memoria y `--time` de los trabajos son estimaciones; los `--gres`, `--partition` y `--account` se pasan por `SBATCH_*`.
- La tubería se probó **en local con datos sintéticos** (etapas `variant`, `sweep`, `select`, `final`); `cache`, `baseline` y `unfrozen` se probaron por separado y con datos sintéticos o en `DRY_RUN`, no encadenadas con datos reales.
- Con 5 semillas la variación del HOTA entre ejecuciones es de ±2–3 puntos (sintético); diferencias menores no son concluyentes.

### Criterio de aceptación del H1
Igualar a `ema` (misma actualización) y no quedar por debajo de `graph` más allá de la desviación entre semillas, con el mismo protocolo (3 semillas, HOTA/DetA/AssA/IDF1/IDSW
globales y en cruces). **Solo se ha verificado con datos sintéticos en CPU**; la comprobación real es en la workstation o el clúster.

### Próximos hitos
H2 predictor de movimiento (GRU con σ) · H3 transformer de grafo · H4 banco histórico · H5 `a_det` (RoIAlign; vector por query en la caché mientras el detector esté
congelado) · H6 ruido de entrenamiento · H7 descongelar el decoder (BN en eval, DDP, control de AP y DetA).
