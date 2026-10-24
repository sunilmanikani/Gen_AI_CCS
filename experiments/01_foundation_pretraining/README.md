# 01 — Foundation Model Pre-Training

Pre-trains a DDPM (Denoising Diffusion Probabilistic Model) on seismic patches extracted from the Sleipner 4D volume. The resulting checkpoint serves as a foundation model for all downstream tasks (RQ5).

## Files

| File | Description |
|---|---|
| `diffusion.ipynb` | Full DDPM training pipeline — data loading, noise schedule, training loop, validation |
| `pretraining.py` | Standalone script version (run via `nohup python pretraining.py &`) |

## Configuration

```python
CONFIG = {
    "VOLUME_PATH": "sliepner.npy",
    "IMG_SIZE": 128,
    "DIFFUSION_STEPS": 1000,
    "BATCH_SIZE": 4,
    "DEVICE": "cuda",
}
```

## Output

Checkpoint saved to `seismic_plume_diffusion_model.pth` (~1.9 GB, not included in repo).
Training epoch images saved to `results/diffusion_epoch_NNN.png`.
