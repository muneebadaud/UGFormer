# Uncertainty-Guided Transformer UIE

A complete PyTorch project for underwater image enhancement with:

- deterministic transformer-based restoration
- pixel-wise uncertainty map
- uncertainty-guided training loss
- paired image dataset loader
- training / evaluation / inference scripts

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

## Notes

- The model is deterministic at inference: no sampling.
- The uncertainty head is used only as a confidence signal.
- You can tune the balance between reconstruction and uncertainty regularization in `train.py`.
