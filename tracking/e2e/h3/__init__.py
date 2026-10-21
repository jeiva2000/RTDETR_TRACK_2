"""H3: módulo de asociación relacional (apariencia + movimiento + relaciones espaciales entre objetos, fusionadas con gates por par track–detección).
Un archivo por componente: appearance_head, motion_predictor, relational_attention, affinity_gates, assignment, track_memory; el resto
(model, losses, noise, train, infer) los ensambla. Se activa con h3.enabled=true (tracking/e2e/config.py); ver docs/E2E.md."""
