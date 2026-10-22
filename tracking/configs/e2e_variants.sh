# Variantes del módulo e2e que lanza la tubería (tracking/scripts/e2e_pipeline.sh). Cada línea: nombre|overrides separados por espacios (-u de tracking.e2e.train).
# Claves del H1 (detector congelado, caché), del H2 (movimiento) y del H3 (asociación relacional: h3.*). Los hitos H4-H6 añadirán aquí sus variantes (banco, a_det, ruido).
# H3 con bucle cerrado: train.ss_max / train.ss_warmup son la proporción de clips que corren con las decisiones y los nacimientos del modelo (0 al empezar, sube linealmente hasta 0.9 en
# 20 épocas; el resto usa el GT) y train.deta_stop=false quita la parada por DetA para entrenar todas las épocas (JOINT_ARGS --epochs, 40 por defecto).
H3_TRAIN="h3.enabled=true train.T=8 train.ss_max=0.9 train.ss_warmup=20 train.deta_stop=false"
E2E_VARIANTS=(
  "h1|"                                          # configuración por defecto (EMA fija alpha 0.1, cabeza de nacimiento desconectada de la apariencia)
  "h1_a02|appearance.alpha_app=0.2"              # EMA más rápida
  "h1_a005|appearance.alpha_app=0.05"            # EMA más lenta
  "h1_pw3|loss.birth_pos_weight=3.0"             # cabeza de nacimiento con más peso en los positivos
  "h2|motion.mode=gru"                           # H2: segundo vector por track (GRU de movimiento con σ), resto igual que h1
  "h2_w01|motion.mode=gru loss.w_motion=0.1"     # H2 con menos peso de la NLL de movimiento
  "h2_m64|motion.mode=gru motion.dim=64 motion.hidden=64"   # H2 con un vector de movimiento más pequeño
  "h3|$H3_TRAIN"                                                              # H3: asociación relacional (apariencia + movimiento + relaciones espaciales, gates por par), clips de 8 frames, con bucle cerrado en el entrenamiento
  "h3_gtonly|h3.enabled=true train.T=8 h3.closed_loop=false"                  # H3 con la receta anterior (nacimientos GT, scheduled sampling por frame, parada por DetA): para comparar
  "h3_nogates|$H3_TRAIN h3.gates=false"                                       # ablación: suma con pesos fijos en lugar de gates por par
  "h3_norel|$H3_TRAIN h3.branches.rel=false"                                  # ablación: sin la rama relacional (sin auto-atención ni atención cruzada)
  "h3_nomot|$H3_TRAIN h3.branches.mot=false"                                  # ablación: sin la rama de movimiento (sin GRU; caja predicha = última observada)
  "h3_noapp|$H3_TRAIN h3.branches.app=false"                                  # ablación: sin la rama de apariencia
  "h3_gate|$H3_TRAIN h3.memory.app_update=gate"                               # actualización de la apariencia con gate aprendido en lugar de EMA
  "h3_edges|$H3_TRAIN h3.memory.edges=true"                                   # fase 2: memoria de aristas (vector por pareja de tracks, GRU) como sesgo extra en la atención de tracks
)
