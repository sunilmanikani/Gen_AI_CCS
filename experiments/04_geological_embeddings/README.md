# 04 — Geological Embeddings (RQ3)

Extracts latent geological representations from seismic patches by passing them through the pre-trained diffusion model at a fixed timestep (t=100). These embeddings capture subsurface structural patterns without supervised labels.

## Files

| File | Description |
|---|---|
| `attributes_embeddings.ipynb` | Full embedding extraction and clustering analysis |
| `embeddings.py` | Standalone embedding extraction script |
| `working/embeddings.ipynb` | Work-in-progress experiments |

## Approach

The pre-trained foundation model (`seismic_plume_diffusion_model.pth`) is used as a frozen feature extractor. Patches are noised to timestep t=100 and the intermediate UNet activations are extracted as 128-dimensional embeddings.

## Key Result

PCA + clustering on diffusion embeddings achieves a **silhouette score 3.52× higher** than a 16× larger convolutional autoencoder — demonstrating the geological structure encoded in diffusion latent space.
