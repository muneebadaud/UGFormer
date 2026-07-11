from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.model import UncertaintyGuidedTransformerUIE
from src.utils import ensure_dir, list_image_files, load_rgb, pil_to_tensor, tensor_to_pil

def save_reliability_map(log_var: torch.Tensor, enhanced_pil: Image.Image,
                          out_dir: Path, stem: str):
    """
    Reliability map r̂ = normalise(exp(−log σ²)) = normalise(1/σ²)
    HIGH value (green) = model is confident → trust the enhancement here
    LOW value  (red)   = model is uncertain → treat with caution
    This is the XAI output. It is the INVERSE of the uncertainty heatmap.
    """
    logv = log_var.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)

    # reliability = exp(−log_var) = 1/σ²
    # note the MINUS sign — this is the key difference from uncertainty
    reliability = np.exp(-logv)

    # save raw values for downstream use (e.g. robotics, annotation weighting)
    np.save(out_dir / f"{stem}_reliability.npy", reliability)

    # normalise for visualisation
    r_vis = robust_norm(reliability, p_low=2.0, p_high=98.0)

    # RdYlGn: red=low trust, yellow=medium, green=high trust
    plt.imsave(out_dir / f"{stem}_reliability_map.png", r_vis, cmap="RdYlGn")

    # overlay on ENHANCED image (not input — reliability is about the OUTPUT)
    enhanced = np.array(enhanced_pil).astype(np.float32) / 255.0
    heat = plt.cm.RdYlGn(r_vis)[..., :3]
    overlay = 0.55 * enhanced + 0.45 * heat
    overlay = (overlay * 255.0).clip(0, 255).astype("uint8")
    Image.fromarray(overlay).save(out_dir / f"{stem}_reliability_overlay.png")

    print(f"{stem} | reliability min={reliability.min():.4f}, "
          f"max={reliability.max():.4f}, mean={reliability.mean():.4f}")

def calc_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    mse = F.mse_loss(pred, target, reduction="mean").item()
    if mse < eps:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def calc_ssim(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    # pred/target: [1, 3, H, W], range [0, 1]
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = pred.mean(dim=(-1, -2), keepdim=True)
    mu_y = target.mean(dim=(-1, -2), keepdim=True)

    sigma_x = ((pred - mu_x) ** 2).mean(dim=(-1, -2), keepdim=True)
    sigma_y = ((target - mu_y) ** 2).mean(dim=(-1, -2), keepdim=True)
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean(dim=(-1, -2), keepdim=True)

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2) + eps
    )
    return ssim_map.mean().item()


def robust_norm(x: np.ndarray, p_low: float = 2.0, p_high: float = 98.0, eps: float = 1e-8) -> np.ndarray:
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + eps)


def save_uncertainty_visuals(log_var: torch.Tensor, input_pil: Image.Image, out_dir: Path, stem: str):
    """
    log_var: [1,1,H,W]
    We visualize uncertainty as exp(log_var) for positivity.
    """
    logv = log_var.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    unc = np.exp(logv)

    # raw log-variance
    np.save(out_dir / f"{stem}_logvar.npy", logv)
    # raw uncertainty
    np.save(out_dir / f"{stem}_uncertainty.npy", unc)

    # visible grayscale / heatmap from uncertainty
    unc_vis = robust_norm(unc, p_low=2.0, p_high=98.0)
    unc_vis = np.power(unc_vis, 0.5)  # gamma boost for visibility

    gray = (unc_vis * 255.0).clip(0, 255).astype("uint8")
    Image.fromarray(gray).save(out_dir / f"{stem}_gray.png")

    plt.imsave(out_dir / f"{stem}_heatmap.png", unc_vis, cmap="magma")

    # overlay on input
    img = np.array(input_pil).astype(np.float32) / 255.0
    heat = plt.cm.magma(unc_vis)[..., :3]
    alpha = 0.6
    overlay = (1.0 - alpha) * img + alpha * heat
    overlay = (overlay * 255.0).clip(0, 255).astype("uint8")
    Image.fromarray(overlay).save(out_dir / f"{stem}_overlay.png")

    print(f"{stem} | log_var min={logv.min():.6f}, max={logv.max():.6f}, mean={logv.mean():.6f}")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Folder with input images")
    parser.add_argument("--gt_root", type=str, default="", help="Folder with ground-truth images")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--save_uncertainty", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = ensure_dir(args.output_dir)
    img_out = ensure_dir(output_dir / "enhanced")
    unc_out = ensure_dir(output_dir / "uncertainty")

    model = UncertaintyGuidedTransformerUIE(dim=args.dim).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    files = list_image_files(args.data_root)
    if not files:
        raise RuntimeError(f"No images found in {args.data_root}")

    use_gt = bool(args.gt_root)
    psnr_sum = 0.0
    ssim_sum = 0.0
    n_metrics = 0

    for fp in tqdm(files, desc="Infer"):
        input_pil = load_rgb(fp)
        x_t = pil_to_tensor(input_pil).unsqueeze(0).to(device)

        pred, log_var = model(x_t)
        pred = pred.clamp(0.0, 1.0)

        tensor_to_pil(pred).save(img_out / fp.name)

        if args.save_uncertainty:
            save_uncertainty_visuals(log_var, input_pil, unc_out, fp.stem)
            # XAI reliability map — pass ENHANCED image, not input
            enhanced_pil = tensor_to_pil(pred)
            save_reliability_map(log_var, enhanced_pil, unc_out, fp.stem)

        if use_gt:
            gt_path = Path(args.gt_root) / fp.name
            if not gt_path.exists():
                print(f"Warning: missing GT for {fp.name}, skipping metrics")
                continue

            gt_pil = load_rgb(gt_path)
            gt_t = pil_to_tensor(gt_pil).unsqueeze(0).to(device)

            if pred.shape[-2:] != gt_t.shape[-2:]:
                pred_m = F.interpolate(pred, size=gt_t.shape[-2:], mode="bilinear", align_corners=False)
            else:
                pred_m = pred

            psnr_val = calc_psnr(pred_m, gt_t)
            ssim_val = calc_ssim(pred_m, gt_t)

            psnr_sum += psnr_val
            ssim_sum += ssim_val
            n_metrics += 1

    print(f"Saved outputs to: {output_dir}")

    if use_gt and n_metrics > 0:
        print(f"Average PSNR: {psnr_sum / n_metrics:.4f}")
        print(f"Average SSIM: {ssim_sum / n_metrics:.4f}")


if __name__ == "__main__":
    main()