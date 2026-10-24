# Gen_AI_CCS — Generative AI for Carbon Capture & Sequestration

Experimental codebase for applying **diffusion-based generative AI** to seismic monitoring in Carbon Capture & Sequestration (CCS) projects.

All experiments use the [Sleipner 4D benchmark dataset](https://co2datashare.org/dataset/sleipner-2019-benchmark-model) — the world's most complete CCS seismic record (Utsira Formation, Norway, CO₂ injection since 1994).

---

## Research Questions

| # | Question | Experiment |
|---|---|---|
| RQ1 | Can diffusion models improve seismic image quality? | `01_foundation_pretraining`, `02_super_resolution` |
| RQ2 | Can AI reconstruct missing seismic traces? | `03_trace_interpolation` |
| RQ3 | Can diffusion models learn geological embeddings? | `04_geological_embeddings` |
| RQ4 | How does diffusion compare to GAN / EDSR / ZSSR? | `02_super_resolution`, `03_trace_interpolation` |
| RQ5 | Can a single pre-trained model serve all tasks? | `01_foundation_pretraining`, `05_uncertainty_quantification` |

---

## Repository Structure

```
Gen_AI_CCS/
├── core/                          # Shared neural network library (rpure.py)
├── experiments/
│   ├── 01_foundation_pretraining/ # DDPM pre-training on seismic patches
│   ├── 02_super_resolution/       # DDPM-SR, EDSR, ZSSR comparison
│   ├── 03_trace_interpolation/    # GAN & DDPM missing trace reconstruction
│   ├── 04_geological_embeddings/  # Diffusion-based geological embeddings
│   └── 05_uncertainty_quantification/ # Zero-shot uplift & uncertainty maps
├── evaluation/                    # SSIM, PSNR, RMS, semblance metrics
└── utils/                         # SEGY ingestion, inference utilities
```

---

## Dataset

The seismic volumes are **not included** in this repository due to file size (~466 MB each).

Download from: **[CO2DataShare — Sleipner Benchmark](https://co2datashare.org/dataset/sleipner-2019-benchmark-model)**

Expected files (place in project root):
```
sliepner.npy              # shape (249, 468, 1001), float32
04p07ful_Realized_1.npy   # same shape, second vintage
```

---

## Setup

```bash
pip install -r requirements.txt
```

GPU (CUDA) required for training. All scripts auto-detect device:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

Random seed: `42` (notebooks) / `1` (interpolation scripts).

---

## Core Library — `core/rpure.py`

Shared neural network components used across all experiments:

| Class | Description |
|---|---|
| `ResUNet++` | Primary generator backbone — residual blocks + skip connections |
| `Discriminator` | PatchGAN-style discriminator for adversarial training |
| `GenConvBlock` | Conv block with Instance Norm, ELU, residual connection |
| `SEBlock` | Squeeze-and-Excitation channel attention |
| `blockUNet` | U-Net building block |
| `read_npy()` | Seismic volume loader with optional memory mapping |

---

## Key Results

| Task | Metric | Result |
|---|---|---|
| Seismic enhancement | PSNR improvement | +5.19 dB over bicubic baseline |
| Seismic enhancement | SSIM | 0.72 → 0.92 |
| Trace interpolation | SSIM (CO₂ plume zone) | 7× advantage over bilinear |
| Geological embeddings | Silhouette score | 3.52× better than 16× larger autoencoder |
| Uncertainty maps | Correlation with amplitude heterogeneity | r = 0.738 |
| Full-volume uncertainty scan | Runtime | 42 seconds (single GPU) |

---

## Dependencies

```
torch >= 2.4.1
numpy
matplotlib
tqdm
scikit-image
scipy
```

---

## Dataset Citation

> Equinor / CO2DataShare. *Sleipner 2019 Benchmark Dataset*. Available at https://co2datashare.org

## License

Code: MIT License. Dataset: subject to CO2DataShare terms of use.
