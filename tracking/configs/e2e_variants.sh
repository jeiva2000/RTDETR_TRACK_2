# Variantes del módulo e2e que lanza la tubería (tracking/scripts/e2e_pipeline.sh). Cada línea: nombre|overrides separados por espacios (-u de tracking.e2e.train).
# Solo existen las claves del H1 (detector congelado, caché). Los hitos H2-H6 añadirán aquí sus variantes (movimiento, grafo, banco, a_det, ruido).
E2E_VARIANTS=(
  "h1|"                                          # configuración por defecto (EMA fija alpha 0.1, cabeza de nacimiento desconectada de la apariencia)
  "h1_a02|appearance.alpha_app=0.2"              # EMA más rápida
  "h1_a005|appearance.alpha_app=0.05"            # EMA más lenta
  "h1_pw3|loss.birth_pos_weight=3.0"             # cabeza de nacimiento con más peso en los positivos
)
