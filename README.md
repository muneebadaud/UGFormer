# Uncertainty-Guided Transformer UIE

## Description
This repository contains the official PyTorch implementation of underwater image enhancement with: transformer-based restoration, pixel-wise uncertainty map and uncertainty-guided training loss.

**UGFormer** jointly predicts:

- A **restored RGB image**.
- A **pixel-wise spatial confidence map**.

The key contributions are:

1. A **Reliability Refine** module using per-pixel aleatoric uncertainty estimates as spatial gates for feature-level refinement.
2. ⁠An **uncertainty-guided training objective** that jointly supervises image reconstruction quality and the calibration of the predicted confidence map.
3. ⁠An **intrinsic explainable output** in the form of a pixel-aligned reliability map.

On standard benchmarks the model achieves **28.37 dB PSNR** and **0.98 SSIM**.

## 1. Model Structure

Overview of the proposed architecture:

<img width="1770" height="691" alt="fig1" src="https://github.com/user-attachments/assets/d1b5f630-6d7d-49b1-9945-21130d8d2c39" />

The model follows a hierarchical encoder-decoder design. The encoder applies transformer blocks at three spatial scales with stride-2 downsampling. The decoder mirrors this with PixelShuffle upsampling and skip-connection fusion.

## Dataset Information

The model is trained and evaluated on paired underwater image datasets. Supported benchmarks:

We used [[LSUI]](https://lintaopeng.github.io/_pages/UIE%20Project%20Page.html) Large-Scale Underwater Image dataset 4500 paied images dataset for the training and 504 paired test images. Available for download at: https://drive.google.com/drive/folders/14BWwtYWCRjWob-jf7Rloj2hSND_oPQdr?usp=sharing

We have also reported on C60 test set of [[UIEB]](https://li-chongyi.github.io/proj_benchmark.html) Underwater Image Enhancement Benchmark. Available of download at: https://drive.google.com/file/d/1Ew_r83nXzVk0hlkfuomWqsAIxuq6kaN4/view

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

- [LSUI dataset](https://lintaopeng.github.io/_pages/UIE%20Project%20Page.html) — Large-Scale Underwater Image dataset
- [UIEB benchmark](https://li-chongyi.github.io/proj_benchmark.html) — An Underwater Image Enhancement Benchmark Dataset and Beyond (TIP 2020)

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for more information.

## Citation

If you use this code or the pre-trained model, please cite:

```bibtex
@article{ugformer2025,
  title   = {Uncertainty Guided Spatial Transformer for Underwater Image Enhancemen},
  author  = {[Waseem Iqbal, Dawood Al Abri, Muneeba Daud, Hammad Afzal, Faisal Iradat, and Ahmad Cheema]},
  journal = {[Venue]},
  year    = {2026}
}
```

