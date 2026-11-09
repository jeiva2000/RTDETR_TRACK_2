"""TrackHeads (etapa 1): una cabeza de atención por track, generada por una hipernetwork, sobre un RT-DETRv2 congelado.
La cabeza de cada objeto del frame t busca ese objeto en la memoria del encoder del frame t+k y predice su caja y si sigue presente.
Ver model.py."""
from tracking.trackheads.model import HeadGenerator, TrackHeads, sine_2d, token_centers

__all__ = ['HeadGenerator', 'TrackHeads', 'sine_2d', 'token_centers']
