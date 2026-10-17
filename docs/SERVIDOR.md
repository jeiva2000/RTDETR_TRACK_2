# Lanzar en el servidor: variante `hist` del módulo de asociación

Todo lo de abajo está **sin ejecutar en el clúster**: los scripts se probaron en local con datos sintéticos. Hay cosas que dependen de
vuestra configuración de SLURM y hay que ajustarlas antes de lanzar (ver "Lo que hay que revisar").

## 0. Preparación (una vez)
```bash
git clone git@github.com:jeiva2000/RTDETR_TRACK_2.git && cd RTDETR_TRACK_2      # o git pull
cp setup_env/paths.hpc.env.example setup_env/paths.hpc.env && nano setup_env/paths.hpc.env   # rutas ABSOLUTAS
mkdir -p logs                                                                    # sbatch falla si no existe
```
`paths.hpc.env` define `PROJECT_ROOT`, `DATA_ROOT`, `CKPT_DIR`, `OUTPUT_ROOT`, `SEED` y `CONDA_ENV`. `DANCETRACK_DIR` se deriva de
`DATA_ROOT` (`$DATA_ROOT/DanceTrack`, con `train1/ train2/ val/`), o se declara en ese archivo si está en otro sitio.

## 1. Entorno
```bash
# si los entornos de conda van en otra carpeta: conda config --add envs_dirs ~/work/envs
bash setup_env/setup_hpc.sh                  # carga GCC/12.2.0 y CUDA/11.8.0, crea el entorno e instala torch (cu118) + numpy<2
conda activate $CONDA_ENV && pip install -r tracking/requirements.txt      # filterpy, lap, scipy
```

## 2. Datos y detector
- **DanceTrack** en `$DANCETRACK_DIR` (`bash setup_env/download_dancetrack.sh hpc` si no está).
- **Checkpoint del detector** (`best.pth` de la Fase 3) copiado al servidor (`rsync`); los checkpoints no van en git.
- Las pruebas del módulo usan `PResNet.pretrained=False`: **no hace falta** descargar pesos ImageNet.

## 3. Caché de features con el detector (GPU)
```bash
DET_CKPT=/ruta/detector/best.pth FEAT_DIR=/ruta/features/detector_v1 sbatch slurm/cache_features.sbatch
```
Crea `$FEAT_DIR/val` y `$FEAT_DIR/train` (unos pocos GB). **Hay que repetirlo si cambia el detector.** Comprueba en el log el
`gt_matched_frac` por secuencia (en la workstation: train ≈ 0.998, val ≈ 0.98, mínimo 0.88 en `dancetrack0018`).

## 4. Baseline con el mismo detector (para la tabla)
```bash
DET_CKPT=/ruta/detector/best.pth RUN_NAME=baseline_v1 sbatch slurm/baseline.sbatch
```
Deja `$OUTPUT_ROOT/baseline_v1/{bytetrack,ocsort}.json`.

## 5. Entrenar y evaluar la variante nueva con 3 semillas
```bash
export FEAT_DIR=/ruta/features/detector_v1
VARIANT=hist  BASE=assoc_hist_A EXTRA="--burn-in 10"                 sbatch slurm/seeds_assoc.sbatch   # afinidad A
VARIANT=hist  BASE=assoc_hist_B EXTRA="--burn-in 10 --hist-aff max"  sbatch slurm/seeds_assoc.sbatch   # afinidad B
VARIANT=graph BASE=assoc_graph                                       sbatch slurm/seeds_assoc.sbatch   # referencia con las mismas semillas
```
Cada comando lanza un array de 3 tareas (semillas 1, 2 y 3). Cada tarea entrena, corre el tracker online sobre las 25 secuencias de val
y evalúa con TrackEval (global y cruces). Resultado en `$OUTPUT_ROOT/<BASE>_s<semilla>/` (`best.pth`, `results/`, `eval.json`, `log.txt`).

## 6. Tabla final (media ± desviación)
```bash
BASELINE_JSONS="bytetrack=$OUTPUT_ROOT/baseline_v1/bytetrack.json ocsort=$OUTPUT_ROOT/baseline_v1/ocsort.json" \
  bash tracking/scripts/aggregate_seeds.sh hpc assoc_hist_A assoc_hist_B assoc_graph        # -> $OUTPUT_ROOT/seeds_table.md
```
Diagnóstico de errores por tipo de decisión y visibilidad (con la caché de val):
```bash
python -m tracking.assoc.diagnose decisions --ckpt $OUTPUT_ROOT/assoc_hist_A_s1/best.pth --cache $FEAT_DIR/val --samples 1000
```

## Lo que hay que revisar antes de lanzar
1. **`#SBATCH` de cada archivo de `slurm/`:** `--gres=gpu:1` es genérico (en el clúster puede ser `gpu:a100:1` o `gpu:1` + `--constraint`),
   y faltan `--partition` y `--account`. No los conozco.
2. **Tiempos y memoria (`--time`, `--mem`):** son estimaciones sin medir. En la workstation `graph` tardaba ~2 min por época con T = 10;
   `hist` con burn-in 10 procesa el doble de frames por clip, y se espera más. El entrenamiento carga las cachés enteras en RAM.
3. **Umbral de nacimiento:** 0.6 por defecto (`BIRTH_THRESH`). El barrido hacia 0.7–0.9 sigue pendiente: antes de comparar en serio,
   conviene fijar el óptimo para todos los métodos con la misma regla.
4. **Semilla:** en el array es `SLURM_ARRAY_TASK_ID` (1, 2, 3); fija e independiente del `SEED` de `paths.hpc.env`.
5. **Variante `hist`:** probada solo con datos sintéticos. Allí no mejoró la re-identificación tras ausencias largas y el
   resultado entre semillas varió mucho; en DanceTrack real es una prueba abierta.
6. **Logs:** `logs/<job>_<id>.out/.err` (hay que crear `logs/`). Los de entrenamiento también van a `$OUTPUT_ROOT/<RUN_NAME>.log`.

---

# Fase 7: fine-tuning conjunto con la variante `hist` (detector descongelado en el decoder)

Entrena **a la vez** el decoder del detector y el módulo de asociación `hist`, con extracción en línea (sin caché). Backbone y encoder
quedan congelados. Pérdida = detección completa de RT-DETRv2 (con denoising) sobre los frames principales de cada clip + `λ` ×
(asociación + visibilidad), con gradiente hacia el decoder a través de los embeddings. Los primeros `burn-in` frames de cada clip se
extraen en modo eval y sin gradiente (llenan estado e historial; no entran en la pérdida de detección). `lr`: decoder 2e-5, módulo 1e-3.

## Dos modos de inicialización del módulo
- **Desde cero (sin `ASSOC_CKPT`):** el módulo se crea con los argumentos de la línea de comandos (`--update hist`, `--hist-aff max`, `--burn-in 10`, ...) y se
  entrena junto al decoder, partiendo **solo del checkpoint del detector**. Por defecto, los primeros **1000 pasos solo aprende el módulo** (decoder a `lr` 0,
  `--det-freeze-steps`) y el scheduled sampling sube de 0 a 0.5 en 3 épocas (`--ss-warmup`), para que un módulo aleatorio no empuje al decoder con gradientes sin sentido.
  Esos valores son una elección mía, sin validar.
- **Desde un checkpoint de la Fase 6 (`ASSOC_CKPT=...`):** el módulo hereda configuración y pesos; sin periodo de solo-módulo.

## Requisitos
1. El detector (`best.pth`) en el servidor. Para el modo (b), además un módulo ya entrenado en la Fase 6 (p. ej. `assoc_hist_A_s1/best.pth`).
2. `DANCETRACK_DIR` con `train1/ train2/ val/` (el entrenamiento lee imágenes y GT directamente).
3. Una GPU grande (A100). **Memoria sin medir:** con 4 clips por paso son 4 × T frames con gradiente por el decoder; si falta, baja `--clips-per-step` o `--T`.

## Lanzar (un entrenamiento, módulo desde cero)
```bash
cd ~/work/RTDETR_TRACK_2 && git pull && mkdir -p logs
DET_CKPT=/ruta/detector/best.pth RUN_NAME=joint_hist_scratch \
  EXTRA="--update hist --burn-in 10 --clips-per-step 4 --samples-per-epoch 1000 --epochs 40 --deta-stride 4" sbatch slurm/train_joint.sbatch
```
(La variante B de afinidad: añade `--hist-aff max` a `EXTRA`.) Con 4 clips por paso, 1000 clips por época son 250 pasos; 40 épocas son 40 000 clips, el mismo orden que la
Fase 6 (2000 × 20). `--deta-stride 4` acelera el control de DetA de cada época (menos frames); el DetA inicial se mide con el mismo stride.
Con checkpoint de la Fase 6: `ASSOC_CKPT=$OUTPUT_ROOT/assoc_hist_A_s1/best.pth` y `EXTRA="--samples-per-epoch 400 --epochs 15"`.
Los valores por defecto: `λ_assoc` = 10, `--lr-det 2e-5`, tolerancia de DetA 0.5 puntos (umbral 0.6).

## Recomendado: barrido de λ (4 entrenamientos en paralelo, una GPU cada uno)
```bash
DET_CKPT=/ruta/detector/best.pth BASE=joint_hist_scratch \
  EXTRA="--update hist --burn-in 10 --clips-per-step 4 --samples-per-epoch 1000 --epochs 40 --deta-stride 4" sbatch slurm/joint_lambda_sweep.sbatch
```
Lanza `λ` = 0, 1, 3 y 10. **`λ` = 0 es el control:** si el DetA baja igualmente, la causa no es la asociación. Resultados en `$OUTPUT_ROOT/<BASE>_l<λ>/`.
Mira `log.txt` (una línea JSON por época: `DetA`, `DetA_by_thr`, `val_acc`, `val_vis_acc`) y la línea final de `train.log`.

## Corrección importante respecto a la primera prueba de la Fase 7
Las capas BatchNorm del decoder (`input_proj`) se actualizaban en cada paso con lotes de frames de un mismo clip (muy correlacionados) y usaban estadísticas
del lote en lugar de las guardadas. Ahora quedan **siempre en modo eval** durante el ajuste (comprobado: con el decoder congelado el detector queda idéntico bit a bit).
Es una explicación plausible, **no demostrada**, de la caída de DetA de la primera prueba; el control con `λ` = 0 lo dirá.

## Control de parada
Si el DetA (umbral 0.6) cae más de 0.5 puntos por debajo del inicial, el entrenamiento **informa y se detiene**. `*_best.pth` es la mejor
`val_acc` entre las épocas válidas; si ninguna lo fue no se crean, y el punto de partida sigue siendo lo mejor. `*_last.pth` se guarda siempre.

## Evaluar el resultado (caché de val con el detector ajustado + baselines con ese detector + tabla)
```bash
RUN_NAME=joint_hist_l3 bash tracking/scripts/eval_joint.sh hpc            # usa detector_best.pth y assoc_best.pth; WHICH=last para los de la última época
```
Para añadir la fila de la Fase 6 con el detector congelado: `EXTRA_JSONS="hist_congelado=$OUTPUT_ROOT/assoc_hist_A_s1/eval.json"`.

## Lo que no está validado
- Con la variante `hist` solo hay pruebas sintéticas (CPU); en DanceTrack real no se ha ejecutado ni la Fase 6 ni la 7 con ella.
- La primera prueba de la Fase 7 (variante `graph`, `λ` = 10, 1 época) **empeoró** DetA (−1.07), `val_acc` y el tracking (HOTA −5.1).
  El experimento de control (`λ` = 0) no tiene resultados todavía. Los valores de `λ`, `--lr-det` y la tolerancia son elecciones sin ajustar.
- Tiempos y memoria de los `.sbatch` sin medir.
