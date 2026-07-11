# Usage:
#   python calibration_analysis.py \
#     --data_root /path/to/LSUI/input \
#     --gt_root   /path/to/LSUI/GT \
#     --checkpoint checkpoints/best.pth \
#     --variant_name "Full UGFormer (F)"


from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm
from PIL import Image

from src.model import UncertaintyGuidedTransformerUIE
from src.utils import list_image_files, load_rgb, pil_to_tensor


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",    required=True)
    parser.add_argument("--gt_root",      required=True)
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--dim",          type=int, default=48)
    parser.add_argument("--variant_name", default="Model")
    # Sample pixels to keep memory manageable (None = use ALL pixels)
    parser.add_argument("--max_pixels",   type=int, default=5_000_000)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Load model ─────────────────────────────────────────────────────────────
    model = UncertaintyGuidedTransformerUIE(dim=args.dim).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    # ── Collect pixel-level data ───────────────────────────────────────────────
    # Store flattened arrays across ALL validation images
    all_variance   = []   # exp(log σ²) — predicted variance per pixel
    all_sq_error   = []   # (ŷ − y)²    — actual squared error per pixel

    # Also track per-image correlations for reporting mean ± std
    per_image_rho  = []

    files = list_image_files(args.data_root)
    print(f"\n{'='*60}")
    print(f"Variant: {args.variant_name}")
    print(f"Images:  {len(files)}")
    print(f"{'='*60}")

    for fp in tqdm(files, desc="Computing calibration"):
        gt_path = Path(args.gt_root) / fp.name
        if not gt_path.exists():
            continue

        # Forward pass
        x  = pil_to_tensor(load_rgb(fp)).unsqueeze(0).to(device)
        y  = pil_to_tensor(load_rgb(gt_path)).unsqueeze(0).to(device)

        pred, log_var = model(x)
        pred = pred.clamp(0.0, 1.0)

        # Align spatial sizes if needed
        if pred.shape[-2:] != y.shape[-2:]:
            pred = F.interpolate(pred, size=y.shape[-2:],
                                 mode="bilinear", align_corners=False)
            log_var = F.interpolate(log_var, size=y.shape[-2:],
                                    mode="bilinear", align_corners=False)

        # ── Per-pixel quantities ───────────────────────────────────────────────
        # Predicted variance: exp(log σ²), shape (1,1,H,W)
        variance = torch.exp(log_var).squeeze().cpu().numpy().astype(np.float32)

        # Squared reconstruction error averaged across RGB channels, shape (H,W)
        sq_err = ((pred - y) ** 2).mean(dim=1).squeeze().cpu().numpy().astype(np.float32)

        # Per-image Spearman ρ
        rho, pval = stats.spearmanr(variance.ravel(), sq_err.ravel())
        per_image_rho.append(rho)

        # Accumulate for global correlation
        all_variance.append(variance.ravel())
        all_sq_error.append(sq_err.ravel())

    # ── Global Spearman correlation ────────────────────────────────────────────
    all_var = np.concatenate(all_variance)
    all_err = np.concatenate(all_sq_error)

    # Sub-sample if too large for memory
    if args.max_pixels and len(all_var) > args.max_pixels:
        idx = np.random.default_rng(42).choice(
            len(all_var), size=args.max_pixels, replace=False
        )
        all_var = all_var[idx]
        all_err = all_err[idx]
        print(f"Sub-sampled to {args.max_pixels:,} pixels for global correlation")

    global_rho, global_pval = stats.spearmanr(all_var, all_err)

    # ── Results ────────────────────────────────────────────────────────────────
    per_image_rho = np.array(per_image_rho)

    print(f"\n{'─'*60}")
    print(f"CALIBRATION RESULTS — {args.variant_name}")
    print(f"{'─'*60}")
    print(f"Global Spearman ρ:     {global_rho:.4f}  (p = {global_pval:.2e})")
    print(f"Per-image mean ρ:      {per_image_rho.mean():.4f} ± {per_image_rho.std():.4f}")
    print(f"Per-image median ρ:    {np.median(per_image_rho):.4f}")
    print(f"Per-image min / max:   {per_image_rho.min():.4f} / {per_image_rho.max():.4f}")
    print(f"Total pixels used:     {len(all_var):,}")
    print(f"{'─'*60}\n")

    # ── Save results to text file ──────────────────────────────────────────────
    out_txt = Path(args.checkpoint).parent / f"calibration_{args.variant_name.replace(' ','_')}.txt"
    with open(out_txt, "w") as f:
        f.write(f"Variant: {args.variant_name}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Global Spearman rho: {global_rho:.6f}\n")
        f.write(f"Global p-value: {global_pval:.6e}\n")
        f.write(f"Per-image mean rho: {per_image_rho.mean():.6f}\n")
        f.write(f"Per-image std rho: {per_image_rho.std():.6f}\n")
        f.write(f"Per-image median rho: {np.median(per_image_rho):.6f}\n")
        f.write(f"N pixels: {len(all_var)}\n")
    print(f"Saved to: {out_txt}")


if __name__ == "__main__":
    main()