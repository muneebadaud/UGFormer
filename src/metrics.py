from __future__ import annotations

import math
import torch


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse < eps:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)
