from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data import PairedFolderDataset
from src.losses import UncertaintyGuidedLoss
from src.metrics import psnr
from src.model import UncertaintyGuidedTransformerUIE
from src.utils import ensure_dir, seed_everything


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_psnr = 0.0
    n = 0
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred, sigma = model(x)
        pred = pred.clamp(0.0, 1.0)
        total_psnr += psnr(pred, y) * x.size(0)
        n += x.size(0)
    return total_psnr / max(n, 1)


def save_checkpoint(path, model, optimizer, epoch, best_psnr, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_psnr": best_psnr,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_name", type=str, default="best.pth")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_tv", type=float, default=0.01)
    parser.add_argument("--dim", type=int, default=48)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = PairedFolderDataset(args.data_root, split="train", img_size=args.img_size)
    val_len = max(1, int(len(dataset) * args.val_ratio))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(
        dataset, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = UncertaintyGuidedTransformerUIE(dim=args.dim).to(device)
    criterion = UncertaintyGuidedLoss(
    lambda_recon=args.lambda_recon,
    lambda_tv=args.lambda_tv,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    save_dir = ensure_dir(args.save_dir)
    ckpt_path = save_dir / args.save_name

    start_epoch = 1
    best_val = -1.0

    if args.resume is not None and Path(args.resume).is_file():
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        # Restore training state
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val = checkpoint.get("best_psnr", -1.0)

        print(f"Resumed from epoch {start_epoch - 1}, best PSNR={best_val:.3f}")

    print(f"Device: {device}")
    print(f"Train: {len(train_set)} | Val: {len(val_set)}")
    print(f"Checkpoint: {ckpt_path}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for x, y, _ in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                pred, log_var = model(x)
                loss, logs = criterion(pred, y, log_var)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += logs["loss_total"].item()
            pbar.set_postfix(loss=f"{logs['loss_total'].item():.4f}", psnr=f"{psnr(pred.clamp(0,1), y):.2f}")

        val_psnr = validate(model, val_loader, device)
        elapsed = time.time() - t0
        avg_train = running / max(len(train_loader), 1)

        print(
            f"Epoch {epoch:03d} | val_psnr={val_psnr:.3f} | "
            f"time={elapsed:.1f}s"
        )

        if val_psnr > best_val:
            best_val = val_psnr
            save_checkpoint(ckpt_path, model, optimizer, epoch, best_val, args)
            print(f"Saved new best checkpoint: {ckpt_path} (PSNR={best_val:.3f})")

    print(f"Done. Best validation PSNR: {best_val:.3f}")


if __name__ == "__main__":
    main()
