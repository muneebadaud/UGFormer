# Uncertainty-Guided Transformer UIE

## Description
This repository contains the official PyTorch implementation of underwater image enhancement with: transformer-based restoration, pixel-wise uncertainty map and uncertainty-guided training loss.

**UGFormer** jointly predicts:

- A **restored RGB image**.
- A **pixel-wise spatial confidence map**.

The key contributions are:

1. A **Reliability Refine** module using per-pixel aleatoric uncertainty estimates as spatial gates for feature-level refinement.
2. ⁠An **uncertainty-guided training objective** that jointly supervises image reconstruction quality and the calibration of the predicted         confidence map.
3. ⁠An **intrinsic explainable output** in the form of a pixel-aligned reliability map.

On standard benchmarks the model achieves **28.37 dB PSNR** and **0.98 SSIM**.

## 1. Model Structure

Overview of the proposed architecture:

![fig1.pdf](https://github.com/user-attachments/files/30385046/fig1.pdf)

The model follows a hierarchical encoder-decoder design. The encoder applies transformer blocks at three spatial scales with stride-2 downsampling. The decoder mirrors this with PixelShuffle upsampling and skip-connection fusion.

## Dataset Information

The model is trained and evaluated on paired underwater image datasets. Supported benchmarks:

- **LSUI** — Large-Scale Underwater Image dataset (3,600 train / 900 val pairs). [[Download]](https://lintaopeng.github.io/_pages/UIE%20Project%20Page.html)
- **UIEB** — Underwater Image Enhancement Benchmark (890 reference images). [[Download]](https://li-chongyi.github.io/proj_benchmark.html)

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
## 4. Code Information

> **[Google Drive — best.pth](https://drive.google.com/file/d/1OiaxbLW93HaYUxyn7G_buVgSJEFHY-5d/view?usp=sharing)**

Place the downloaded file in the `checkpoints/` folder:

```
checkpoints/
  best.pth
```

## References

- [UIEB benchmark](https://li-chongyi.github.io/proj_benchmark.html) — An Underwater Image Enhancement Benchmark Dataset and Beyond (TIP 2020)
- [LSUI dataset](https://lintaopeng.github.io/_pages/UIE%20Project%20Page.html) — Large-Scale Underwater Image dataset

  
## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Citation

If you use this code or the pre-trained model, please cite:

```bibtex
@article{ugformer2025,
  title   = {Confidence-Guided Underwater Image Enhancement with a Deterministic Transformer},
  author  = {[Author Names]},
  journal = {[Venue]},
  year    = {2025}
}
```

