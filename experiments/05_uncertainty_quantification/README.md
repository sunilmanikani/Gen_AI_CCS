# 05 — Uncertainty Quantification (RQ1, RQ5)

Generates spatially structured uncertainty maps using Ensemble Uncertainty Quantification (EUQ) — multiple forward passes through the diffusion model with different noise seeds, measuring prediction variance across the seismic volume.

## Files

| File | Description |
|---|---|
| `diffusion_zero_shot_uplifts.ipynb` | Zero-shot diffusion denoising with amplitude, RMS, and semblance uplift metrics |
| `probability_maps.py` | Generates CO₂ plume probability maps from ensemble predictions |
| `uncertainty_overlay.py` | Overlays uncertainty bands on seismic sections for visualisation |

## Key Results

| Metric | Result |
|---|---|
| Uncertainty vs amplitude heterogeneity correlation | r = 0.738 |
| Full-volume uncertainty scan (249 inlines) | 42 seconds on a single GPU |
| Amplitude uplift at epoch 20 | +5.19 dB PSNR |
| RMS uplift after fine-tuning | Consistent across full Sleipner volume |

High uncertainty correctly localises to the CO₂ plume boundaries and fault zones — geologically meaningful without supervision.
