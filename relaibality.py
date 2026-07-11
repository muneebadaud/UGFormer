# Run after inference to produce annotated reliability maps 
# Usage: python annotate_reliability.py --input_dir outputs/uncertainty --output_dir outputs/annotated

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from PIL import Image
from scipy import ndimage


def robust_norm(x, p_low=2.0, p_high=98.0, eps=1e-8):
    lo, hi = np.percentile(x, p_low), np.percentile(x, p_high)
    return (np.clip(x, lo, hi) - lo) / (hi - lo + eps)


def annotate_reliability(
    r_hat: np.ndarray,          
    enhanced_img: np.ndarray,   
    stem: str,
    out_dir: Path,
    trust_thresh: float = 0.8,
    caution_thresh: float = 0.2,
):
    H, W = r_hat.shape

    r_vis = robust_norm(r_hat)


    trust_mask   = r_vis > trust_thresh    # green: trust
    caution_mask = r_vis < caution_thresh  # red: treat with caution

    overlay = enhanced_img.copy()

    # Blend green into trust regions
    green = np.array([0.1, 0.8, 0.1], dtype=np.float32)
    overlay[trust_mask] = 0.55 * overlay[trust_mask] + 0.45 * green

    # Blend red into caution regions
    red = np.array([0.9, 0.1, 0.1], dtype=np.float32)
    overlay[caution_mask] = 0.55 * overlay[caution_mask] + 0.45 * red

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('white')

    axes[0].imshow(enhanced_img.clip(0, 1))
    axes[0].set_title('Enhanced ŷ', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    im = axes[1].imshow(r_vis, cmap='RdYlGn', vmin=0, vmax=1)
    axes[1].set_title('Reliability map r̂', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04,
                 label='r̂ (0 = uncertain, 1 = reliable)')

    axes[1].contour(trust_mask.astype(float),   levels=[0.5],
                    colors=['#00AA00'], linewidths=1.5)
    axes[1].contour(caution_mask.astype(float), levels=[0.5],
                    colors=['#CC0000'], linewidths=1.5)

    axes[1].text(
        0.02, 0.98,
        f'■  r̂ > {trust_thresh:.1f} — Trust',
        transform=axes[1].transAxes,
        fontsize=9, color='#00AA00',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#00AA00')
    )
    axes[1].text(
        0.02, 0.88,
        f'■  r̂ < {caution_thresh:.1f} — Caution',
        transform=axes[1].transAxes,
        fontsize=9, color='#CC0000',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#CC0000')
    )

    for mask, color, label in [
        (trust_mask,   '#00AA00', 'Trust\n(r̂ > 0.8)'),
        (caution_mask, '#CC0000', 'Caution\n(r̂ < 0.2)')
    ]:
        # Label the centroid of the two largest connected components
        labeled, n_comp = ndimage.label(mask)
        if n_comp == 0:
            continue
        sizes = ndimage.sum(mask, labeled, range(1, n_comp + 1))
        top_n = min(2, n_comp)
        top_labels = np.argsort(sizes)[::-1][:top_n] + 1

        for comp_label in top_labels:
            comp_mask = labeled == comp_label
            cy, cx = ndimage.center_of_mass(comp_mask)
            # Only annotate if component is large enough to be meaningful
            if sizes[comp_label - 1] < (H * W * 0.005):
                continue
            axes[1].annotate(
                label,
                xy=(cx, cy),
                fontsize=8, color=color, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          alpha=0.75, edgecolor=color, linewidth=1.2)
            )

    # Panel 3: Overlay on enhanced image
    axes[2].imshow(overlay.clip(0, 1))
    axes[2].set_title('r̂ Overlay on ŷ', fontsize=13, fontweight='bold')
    axes[2].axis('off')

    # Legend patches
    legend_handles = [
        mpatches.Patch(color='#00AA00', alpha=0.7, label=f'r̂ > {trust_thresh:.1f}  — Trust'),
        mpatches.Patch(color='#CC0000', alpha=0.7, label=f'r̂ < {caution_thresh:.1f}  — Caution'),
    ]
    axes[2].legend(
        handles=legend_handles,
        loc='lower left', fontsize=9,
        framealpha=0.85, edgecolor='gray'
    )

    trust_pct   = 100 * trust_mask.mean()
    caution_pct = 100 * caution_mask.mean()
    fig.text(
        0.5, 0.01,
        f'Trust region (r̂ > {trust_thresh:.1f}): {trust_pct:.1f}% of pixels   |   '
        f'Caution region (r̂ < {caution_thresh:.1f}): {caution_pct:.1f}% of pixels   |   '
        f'Mean r̂: {r_vis.mean():.3f}',
        ha='center', fontsize=10, color='#333333'
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = out_dir / f"{stem}_annotated.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"{stem} | trust={trust_pct:.1f}% | caution={caution_pct:.1f}% | mean_r={r_vis.mean():.3f} → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  type=str, required=True,
                        help="Folder containing _reliability.npy and enhanced .png files")
    parser.add_argument("--output_dir", type=str, default="outputs/annotated")
    parser.add_argument("--trust_thresh",   type=float, default=0.8)
    parser.add_argument("--caution_thresh", type=float, default=0.2)
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all reliability .npy files
    rel_files = sorted(input_dir.glob("*_reliability.npy"))
    if not rel_files:
        raise RuntimeError(
            f"No *_reliability.npy files found in {input_dir}.\n"
            "Run infer.py with --save_uncertainty first."
        )

    for rel_path in rel_files:
        stem = rel_path.stem.replace("_reliability", "")

        # Load raw reliability values
        r_hat = np.load(rel_path).astype(np.float32)

        # Load corresponding enhanced image from outputs/enhanced/
        enhanced_dir = input_dir.parent / "enhanced"
        img_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = enhanced_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(f"Warning: no enhanced image found for {stem}, skipping")
            continue

        enhanced = np.array(Image.open(img_path).convert("RGB"),
                            dtype=np.float32) / 255.0

        annotate_reliability(
            r_hat, enhanced, stem, output_dir,
            trust_thresh=args.trust_thresh,
            caution_thresh=args.caution_thresh,
        )

    print(f"\nAnnotated maps saved to: {output_dir}")


if __name__ == "__main__":
    main()