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
