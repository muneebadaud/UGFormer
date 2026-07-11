from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset

from .utils import list_image_files, load_rgb, pil_to_tensor


class PairedFolderDataset(Dataset):


    def __init__(self, root: str | Path, split: str = "train", img_size: Optional[int] = None):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size

        self.input_dir = self.root / "input"
        self.target_dir = self.root / "target"

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Missing input directory: {self.input_dir}")
        if not self.target_dir.exists():
            raise FileNotFoundError(f"Missing target directory: {self.target_dir}")

        in_files = list_image_files(self.input_dir)
        tar_files = list_image_files(self.target_dir)
        tar_map = {p.stem: p for p in tar_files}

        pairs = []
        for ip in in_files:
            tp = tar_map.get(ip.stem)
            if tp is not None:
                pairs.append((ip, tp))

        if not pairs:
            raise RuntimeError(f"No paired files found under {self.input_dir} and {self.target_dir}")

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        in_path, tar_path = self.pairs[idx]
        x = load_rgb(in_path)
        y = load_rgb(tar_path)

        if self.img_size is not None:
            x = x.resize((self.img_size, self.img_size), resample=Image.BICUBIC)
            y = y.resize((self.img_size, self.img_size), resample=Image.BICUBIC)

        return pil_to_tensor(x), pil_to_tensor(y), in_path.name
