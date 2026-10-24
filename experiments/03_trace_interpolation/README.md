# 03 — Missing Trace Interpolation (RQ2, RQ4)

Reconstructs missing or sparsely acquired seismic traces using both GAN-based and DDPM-based approaches. Addresses the challenge of reducing acquisition density requirements in long-term CCS monitoring.

## Files

| File | Description |
|---|---|
| `interpolation.ipynb` | GAN-based interpolation — full training and evaluation pipeline |
| `interpolation.py` | Standalone production script (recommended for long training runs) |
| `ddpm_interp_v8.py` | Latest DDPM-based interpolation — best performance |

## Running

```bash
# Recommended for long runs
nohup python interpolation.py > nohup.out 2>&1 &

# Or headless notebook execution
jupyter nbconvert --to notebook --execute interpolation.ipynb
```

## Key Finding

DDPM interpolation achieves **7× higher SSIM** than bilinear interpolation specifically within the CO₂ plume zone — the region of highest monitoring interest.

## Checkpoints

Saved to `checkpoints_interpolation/` and `models_interpolation/` (not included in repo).
