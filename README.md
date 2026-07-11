# Uncertainty-Guided Transformer UIE

A complete PyTorch project for underwater image enhancement with: deterministic transformer-based restoration, pixel-wise uncertainty map and uncertainty-guided training loss.


## Dataset layout

Place paired images in this structure:

```text
data/
  train/
    input/
      0001.png
      0002.png
    target/
      0001.png
      0002.png
  val/
    input/
    target/
  test/
    input/
    target/   # optional for metrics
```

The loader matches pairs by filename stem.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py \
  --data_root /path/to/data \
  --save_dir checkpoints \
  --device cuda:0 \
  --epochs 200 \
  --batch_size 4 \
  --amp
```

## Inference

```bash
python infer.py \
  --data_root /path/to/data/test/input \
  --checkpoint checkpoints/best.pth \
  --output_dir outputs \
  --device cuda:0 \
  --save_uncertainty
```

Download the checkpoint from https://drive.google.com/file/d/1OiaxbLW93HaYUxyn7G_buVgSJEFHY-5d/view?usp=sharing and place it in checkpoint folder.
