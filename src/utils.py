from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def seed_everything(seed: int = 123) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_image_files(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu().clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse < eps:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def total_variation(x: torch.Tensor) -> torch.Tensor:
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dh + dw
