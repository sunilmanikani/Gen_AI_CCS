# 02 — Super-Resolution (RQ1, RQ4)

2× seismic super-resolution experiments comparing diffusion-based upscaling against EDSR and ZSSR baselines.

## Files

| File | Description |
|---|---|
| `sr_diffusion.ipynb` | DDPM-SR — primary diffusion super-resolution approach |
| `ddpm_sr_v2.py` | Training script for DDPM-SR |
| `ddpm_sr_resume.py` | Resume training from checkpoint |
| `sr_edsr.ipynb` | EDSR (Enhanced Deep Super-Resolution) — comparison baseline |
| `ZSSR.ipynb` | Zero-Shot Super-Resolution — internal learning baseline |
| `sliepner_sr.ipynb` | Sliepner-specific SR experiments with qualitative analysis |

## Results Summary

| Method | PSNR (dB) | SSIM |
|---|---|---|
| Bicubic baseline | — | 0.72 |
| EDSR | +2.1 | 0.81 |
| ZSSR | +1.8 | 0.78 |
| **DDPM-SR (ours)** | **+5.19** | **0.92** |

DDPM-SR shows a 7× performance advantage specifically in the CO₂ plume zone.

## Checkpoints

Saved to `checkpoints/` (not included — too large for git):
- `checkpoints/diffusion_best.pth`
- `checkpoints/edsr_best.pth`
- `checkpoints/ddpm_sr_epoch*.pth`
