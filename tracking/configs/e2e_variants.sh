# Variantes del módulo e2e que lanza la tubería (tracking/scripts/e2e_pipeline.sh). Cada línea: nombre|overrides separados por espacios (-u de tracking.e2e.train).
# Claves del H1 (detector congelado, caché) y del H2 (movimiento). Los hitos H3-H6 añadirán aquí sus variantes (grafo, banco, a_det, ruido).
E2E_VARIANTS=(
  "h1|"                                          # configuración por defecto (EMA fija alpha 0.1, cabeza de nacimiento desconectada de la apariencia)
  "h1_a02|appearance.alpha_app=0.2"              # EMA más rápida
  "h1_a005|appearance.alpha_app=0.05"            # EMA más lenta
  "h1_pw3|loss.birth_pos_weight=3.0"             # cabeza de nacimiento con más peso en los positivos
  "h2|motion.mode=gru"                           # H2: segundo vector por track (GRU de movimiento con σ), resto igual que h1
  "h2_w01|motion.mode=gru loss.w_motion=0.1"     # H2 con menos peso de la NLL de movimiento
  "h2_m64|motion.mode=gru motion.dim=64 motion.hidden=64"   # H2 con un vector de movimiento más pequeño
)
