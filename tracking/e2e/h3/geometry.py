"""Geometría de cajas cxcywh normalizadas, vectorizada por lote (B, N, ...) y válida con N = 0."""
import torch


def to_xyxy(b):
    return torch.stack([b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2, b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2], -1)


def giou(a, b):
    """GIoU en [-1, 1] elemento a elemento con broadcasting: a, b (..., 4) cxcywh -> (...)."""
    a, b = to_xyxy(a), to_xyxy(b)
    area = lambda x: (x[..., 2] - x[..., 0]).clamp(min=0) * (x[..., 3] - x[..., 1]).clamp(min=0)
    inter = (torch.min(a[..., 2], b[..., 2]) - torch.max(a[..., 0], b[..., 0])).clamp(min=0) * (torch.min(a[..., 3], b[..., 3]) - torch.max(a[..., 1], b[..., 1])).clamp(min=0)
    union = area(a) + area(b) - inter + 1e-9
    enc = (torch.max(a[..., 2], b[..., 2]) - torch.min(a[..., 0], b[..., 0])) * (torch.max(a[..., 3], b[..., 3]) - torch.min(a[..., 1], b[..., 1])) + 1e-9
    return inter / union - (enc - union) / enc


def giou_pairwise(a, b):
    """a (B,D,4), b (B,T,4) -> GIoU (B,D,T)."""
    return giou(a[:, :, None], b[:, None])
