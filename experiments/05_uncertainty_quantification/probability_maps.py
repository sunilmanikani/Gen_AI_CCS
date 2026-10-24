#!/usr/bin/env python3
"""
probability_maps.py — Uncertainty quantification via Monte Carlo diffusion sampling.

For each seismic inline, run N stochastic passes through the PDRM at fixed t
with different noise realisations. Pixel-wise variance across passes = uncertainty.

Outputs:
  /datadisk/tupi/tupi/Sliepner/uncertainty_output/uncertainty_panel.png
  /datadisk/tupi/tupi/Sliepner/uncertainty_output/uncertainty_volume.png
  /datadisk/tupi/tupi/Sliepner/uncertainty_output/uncertainty_profile.png
  /datadisk/tupi/tupi/Sliepner/uncertainty_output/uncertainty_histogram.png
  /datadisk/tupi/tupi/Sliepner/uncertainty_output/uncertainty_all.npy
"""

import os, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
BASE     = "/datadisk/tupi/tupi/Sliepner"
DIFF_CKPT = os.path.join(BASE, "embeddings_output/checkpoints/unet_diffusion.pth")
VOLUME_PATH = os.path.join(BASE, "sliepner.npy")
OUT_DIR  = os.path.join(BASE, "uncertainty_output")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE   = 256
BATCH_SIZE = 1       # one inline at a time for MC sampling
SEED       = 42
DIFF_STEPS = 1000
UNET_DIM   = 32
N_MC       = 30      # Monte Carlo samples per inline
EXTRACT_T  = 100     # extraction timestep

TRIM       = 10      # skip first/last N inlines
# Representative inlines to show in detail (spread across the volume)
PANEL_INLINES = [20, 60, 100, 124, 180, 228]   # 6 inlines for panel

DARK  = '#0e0e0e'
PANEL = '#181818'

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Device: {DEVICE}  |  MC samples: {N_MC}  |  t={EXTRACT_T}")

# ── Architecture (exact from pretraining.py) ──────────────────────────────────
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, x):
        device = x.device; half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act  = nn.SiLU()
    def forward(self, x, scale_shift=None):
        x = self.norm(self.proj(x))
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out*2)) if time_emb_dim else None
        self.block1 = Block(dim,     dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb    = self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        return self.block2(h) + self.res_conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale  = dim_head**-0.5; self.heads = heads
        hidden_dim  = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim*3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.heads, -1, h*w), qkv)
        sim  = torch.einsum('b h d i, b h d j -> b h i j', q*self.scale, k)
        attn = sim.softmax(dim=-1)
        out  = torch.einsum('b h i j, b h d j -> b h d i', attn, v)
        return self.to_out(out.reshape(b, -1, h, w)) + x

class UNet(nn.Module):
    def __init__(self, dim=UNET_DIM, out_dim=1, channels=1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim), nn.Linear(dim, dim*4),
            nn.GELU(), nn.Linear(dim*4, dim*4),
        )
        self.init_conv    = nn.Conv2d(channels, dim, 7, padding=3)
        self.downs        = nn.ModuleList([
            ResnetBlock(dim,   dim,   time_emb_dim=dim*4),
            ResnetBlock(dim,   dim*2, time_emb_dim=dim*4),
            ResnetBlock(dim*2, dim*4, time_emb_dim=dim*4),
        ])
        self.down_samples = nn.ModuleList([
            nn.Conv2d(dim,   dim,   4, 2, 1),
            nn.Conv2d(dim*2, dim*2, 4, 2, 1),
            nn.Conv2d(dim*4, dim*4, 4, 2, 1),
        ])
        self.mid_block1 = ResnetBlock(dim*4, dim*4, time_emb_dim=dim*4)
        self.mid_attn   = Attention(dim*4)
        self.mid_block2 = ResnetBlock(dim*4, dim*4, time_emb_dim=dim*4)
        self.ups        = nn.ModuleList([
            ResnetBlock(dim*8, dim*2, time_emb_dim=dim*4),
            ResnetBlock(dim*4, dim,   time_emb_dim=dim*4),
            ResnetBlock(dim*2, dim,   time_emb_dim=dim*4),
        ])
        self.up_samples = nn.ModuleList([
            nn.ConvTranspose2d(dim*4, dim*4, 4, 2, 1),
            nn.ConvTranspose2d(dim*2, dim*2, 4, 2, 1),
            nn.ConvTranspose2d(dim,   dim,   4, 2, 1),
        ])
        self.final_conv = nn.Conv2d(dim, out_dim, 1)

    def forward(self, x, time):
        t      = self.time_mlp(time)
        x      = self.init_conv(x)
        h_vals = []
        for block, sample in zip(self.downs, self.down_samples):
            x = block(x, t); h_vals.append(x); x = sample(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        for block, sample in zip(self.ups, self.up_samples):
            x = sample(x); h = h_vals.pop()
            if x.shape[2:] != h.shape[2:]:
                x = F.interpolate(x, size=h.shape[2:], mode='nearest')
            x = torch.cat((x, h), dim=1)
            x = block(x, t)
        return self.final_conv(x)

# ── Diffusion utils ───────────────────────────────────────────────────────────
class DiffUtils:
    def __init__(self, steps=DIFF_STEPS, device=DEVICE):
        self.steps = steps; self.device = device
        betas     = torch.linspace(0.0001, 0.02, steps).to(device)
        alphas    = 1. - betas
        acp       = torch.cumprod(alphas, dim=0)
        self.sa   = torch.sqrt(acp)
        self.so   = torch.sqrt(1. - acp)
        self.betas = betas

    def q_sample(self, x0, t, noise=None):
        if noise is None: noise = torch.randn_like(x0)
        return self.sa[t].view(-1,1,1,1)*x0 + self.so[t].view(-1,1,1,1)*noise, noise

    def predict_x0(self, x_t, t, pred_noise):
        return (x_t - self.so[t].view(-1,1,1,1)*pred_noise) / self.sa[t].view(-1,1,1,1)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\nLoading PDRM from {DIFF_CKPT}...")
unet = UNet().to(DEVICE)
unet.load_state_dict(
    torch.load(DIFF_CKPT, map_location=DEVICE, weights_only=False)["state_dict"])
unet.eval()
diff = DiffUtils()
print("  Model loaded.")

# ── Load volume ───────────────────────────────────────────────────────────────
print(f"Loading volume...")
vol = np.load(VOLUME_PATH).astype(np.float32)
robust_max = np.percentile(np.abs(vol), 99)
vol = np.clip(vol / robust_max, -1, 1)
N_INLINES = vol.shape[0]
print(f"  shape={vol.shape}  normalised")

def load_inline(idx):
    img = vol[idx]
    img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    img_t = F.interpolate(img_t, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
    return img_t.squeeze(0)          # (1, H, W)

# ── Monte Carlo uncertainty estimation ────────────────────────────────────────
@torch.no_grad()
def mc_uncertainty(img_tensor, n_mc=N_MC, t_val=EXTRACT_T):
    """
    Run N_MC stochastic passes: each adds fresh random noise at t,
    reconstructs x0 via single-step DDPM inversion.
    Returns: mean_recon, std_map (pixel-wise uncertainty), all_recons
    """
    x0    = img_tensor.unsqueeze(0).to(DEVICE)   # (1,1,H,W)
    t_ten = torch.full((1,), t_val, device=DEVICE, dtype=torch.long)
    recons = []
    for _ in range(n_mc):
        noise   = torch.randn_like(x0)
        x_noisy, _ = diff.q_sample(x0, t_ten, noise)
        pred_n  = unet(x_noisy, t_ten)
        x0_hat  = diff.predict_x0(x_noisy, t_ten, pred_n).clamp(-1, 1)
        recons.append(x0_hat.squeeze().cpu().numpy())  # (H, W)

    recons   = np.stack(recons, axis=0)          # (N_MC, H, W)
    mean_r   = recons.mean(axis=0)               # (H, W)
    std_r    = recons.std(axis=0)                # (H, W) — pixel uncertainty
    return mean_r, std_r, recons

# ─────────────────────────────────────────────────────────────────────────────
# PASS 1: Full volume uncertainty scan (mean std per inline)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n── Full volume uncertainty scan ({N_INLINES} inlines, {N_MC} MC each) ──")
vol_mean_unc = []   # mean uncertainty per inline
vol_max_unc  = []   # max uncertainty per inline
vol_p95_unc  = []   # 95th percentile uncertainty per inline

for idx in tqdm(range(N_INLINES), desc="Scanning volume"):
    img_t      = load_inline(idx)
    _, std_map, _ = mc_uncertainty(img_t, n_mc=N_MC, t_val=EXTRACT_T)
    vol_mean_unc.append(float(std_map.mean()))
    vol_max_unc.append(float(std_map.max()))
    vol_p95_unc.append(float(np.percentile(std_map, 95)))

vol_mean_unc = np.array(vol_mean_unc)
vol_max_unc  = np.array(vol_max_unc)
vol_p95_unc  = np.array(vol_p95_unc)
np.save(os.path.join(OUT_DIR, "uncertainty_profile.npy"), vol_mean_unc)
print(f"  Mean uncertainty range: [{vol_mean_unc.min():.4f}, {vol_mean_unc.max():.4f}]")
print(f"  High uncertainty inlines (top 10%): {np.where(vol_mean_unc > np.percentile(vol_mean_unc, 90))[0].tolist()}")

# ─────────────────────────────────────────────────────────────────────────────
# PASS 2: Full MC data for panel inlines
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n── Detailed MC for panel inlines {PANEL_INLINES} ──")
panel_data = {}
for idx in PANEL_INLINES:
    print(f"  Inline {idx}...", end=" ", flush=True)
    img_t = load_inline(idx)
    mean_r, std_r, all_r = mc_uncertainty(img_t, n_mc=N_MC, t_val=EXTRACT_T)
    cv = np.abs(std_r / (np.abs(mean_r) + 1e-6))   # coefficient of variation
    panel_data[idx] = {
        "orig":   img_t.squeeze().numpy(),
        "mean":   mean_r,
        "std":    std_r,
        "cv":     cv,
        "all":    all_r,
        "unc_mean": float(std_r.mean()),
        "unc_p95":  float(np.percentile(std_r, 95)),
    }
    print(f"mean_unc={std_r.mean():.4f}  p95={np.percentile(std_r, 95):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1: Detail panel — 6 inlines × 4 columns
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Plotting uncertainty panel ──")
n_rows = len(PANEL_INLINES)
fig = plt.figure(figsize=(22, n_rows * 3.8), facecolor=DARK)
fig.suptitle(
    f"Monte Carlo Uncertainty Maps  —  PDRM  |  N={N_MC} stochastic passes at t={EXTRACT_T}\n"
    "Column 1: Original seismic  |  Column 2: Mean reconstruction  "
    "|  Column 3: Uncertainty (std)  |  Column 4: Coefficient of variation",
    color='white', fontsize=11, y=1.005,
)

gs = gridspec.GridSpec(n_rows, 4, hspace=0.08, wspace=0.04,
                       left=0.07, right=0.97, top=0.96, bottom=0.04)

# Find global std range for consistent colormap
all_stds = np.concatenate([panel_data[i]["std"].ravel() for i in PANEL_INLINES])
vmin_std, vmax_std = np.percentile(all_stds, 2), np.percentile(all_stds, 98)

col_titles = ["Original\n(sliepner.npy)", "Mean Reconstruction\n(E[x̂₀])",
              "Pixel Uncertainty\n(σ = std across passes)", "Coeff. of Variation\n(σ/|μ|)"]

for row, idx in enumerate(PANEL_INLINES):
    d    = panel_data[idx]
    orig = d["orig"].T
    mean = d["mean"].T
    std  = d["std"].T
    cv   = np.clip(d["cv"].T, 0, 1)

    imgs   = [orig, mean, std, cv]
    cmaps  = ['seismic', 'seismic', 'hot', 'YlOrRd']
    vmins  = [-1, -1, vmin_std, 0]
    vmaxs  = [1,   1, vmax_std, 0.5]

    for col in range(4):
        ax  = fig.add_subplot(gs[row, col])
        ax.set_facecolor(PANEL)
        im  = ax.imshow(imgs[col], cmap=cmaps[col], vmin=vmins[col],
                        vmax=vmaxs[col], aspect='auto', interpolation='nearest')
        ax.axis('off')

        if row == 0:
            ax.set_title(col_titles[col], color='white', fontsize=9, pad=6)

        if col == 0:
            ax.text(-0.02, 0.5, f"Inline {idx:03d}",
                    transform=ax.transAxes, color='#ffcc80',
                    fontsize=8, va='center', ha='right', rotation=90)

        # Annotate uncertainty stats on std and CV columns
        if col == 2:
            ax.text(0.02, 0.97,
                    f"mean σ={d['unc_mean']:.4f}\n95th={d['unc_p95']:.4f}",
                    transform=ax.transAxes, color='white', fontsize=7,
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#111', ec='#555', alpha=0.85))
        if col == 3:
            high_cv_pct = float((d["cv"] > 0.3).mean() * 100)
            ax.text(0.02, 0.97, f"CV>0.3: {high_cv_pct:.1f}%",
                    transform=ax.transAxes, color='white', fontsize=7,
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#111', ec='#555', alpha=0.85))

# Colorbars
for col_idx, (cmap, vmin, vmax, label) in enumerate([
    ('seismic', -1,        1,        'Normalised amplitude'),
    ('seismic', -1,        1,        'Normalised amplitude'),
    ('hot',     vmin_std,  vmax_std, 'Std dev (uncertainty)'),
    ('YlOrRd',  0,         0.5,      'CV  σ/|μ|'),
]):
    cax = fig.add_axes([0.07 + col_idx*0.225, 0.01, 0.18, 0.012])
    sm  = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cb  = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_label(label, color='#ccc', fontsize=7)
    cb.ax.tick_params(colors='#aaa', labelsize=6)

path = os.path.join(OUT_DIR, "uncertainty_panel.png")
plt.savefig(path, dpi=130, bbox_inches='tight', facecolor=DARK)
plt.close(); print(f"  -> {path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2: Uncertainty profile across the volume (all 249 inlines)
# ─────────────────────────────────────────────────────────────────────────────
print("── Plotting uncertainty profile ──")
inline_ids = np.arange(N_INLINES)

fig, axes = plt.subplots(2, 1, figsize=(16, 9), facecolor=DARK, sharex=True)
fig.suptitle(
    f"Uncertainty Profile Across Sleipner Volume  —  {N_INLINES} inlines\n"
    f"N={N_MC} Monte Carlo passes per inline at extraction timestep t={EXTRACT_T}",
    color='white', fontsize=11, y=1.02,
)

# Top: mean and p95 uncertainty
ax = axes[0]; ax.set_facecolor(PANEL)
ax.plot(inline_ids, vol_mean_unc, color='#4fc3f7', lw=1.5, label='Mean pixel σ', alpha=0.9)
ax.fill_between(inline_ids, vol_mean_unc * 0.8, vol_mean_unc * 1.2,
                color='#4fc3f7', alpha=0.15)
ax.plot(inline_ids, vol_p95_unc, color='#ffcc80', lw=1.2, ls='--',
        label='95th-percentile pixel σ', alpha=0.85)

# Mark high-uncertainty zones
thresh = np.percentile(vol_mean_unc, 80)
ax.axhline(thresh, color='#ff7043', lw=0.8, ls=':', alpha=0.7, label=f'80th pct threshold')
high_mask = vol_mean_unc > thresh
ax.fill_between(inline_ids, 0, vol_mean_unc,
                where=high_mask, color='#ff7043', alpha=0.25, label='High uncertainty zones')

# Mark panel inlines
for idx in PANEL_INLINES:
    ax.axvline(idx, color='#a5d6a7', lw=0.8, alpha=0.6)
    ax.text(idx, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.06,
            f' {idx}', color='#a5d6a7', fontsize=7, va='top')

for sp in ax.spines.values(): sp.set_edgecolor('#333')
ax.tick_params(colors='#aaa', labelsize=8)
ax.set_ylabel('Pixel uncertainty (σ)', color='#aaa', fontsize=9)
ax.legend(facecolor='#222', labelcolor='white', edgecolor='#444', fontsize=8, loc='upper right')
ax.set_title('Mean & 95th-percentile pixel uncertainty by inline', color='white', fontsize=9, pad=4)

# Bottom: max uncertainty
ax2 = axes[1]; ax2.set_facecolor(PANEL)
ax2.plot(inline_ids, vol_max_unc, color='#ef9a9a', lw=1.5, label='Max pixel σ')
ax2.fill_between(inline_ids, 0, vol_max_unc, color='#ef9a9a', alpha=0.2)
ax2.axvline(TRIM,          color='white', lw=0.8, ls=':', alpha=0.5)
ax2.axvline(N_INLINES-TRIM, color='white', lw=0.8, ls=':', alpha=0.5)
ax2.text(TRIM + 1, vol_max_unc.max()*0.92, 'trim\nboundary',
         color='white', fontsize=7, alpha=0.6)
for sp in ax2.spines.values(): sp.set_edgecolor('#333')
ax2.tick_params(colors='#aaa', labelsize=8)
ax2.set_ylabel('Max pixel uncertainty (σ)', color='#aaa', fontsize=9)
ax2.set_xlabel('Inline index (lateral position)', color='#aaa', fontsize=9)
ax2.set_title('Maximum pixel uncertainty by inline  (peak uncertainty locations)', color='white', fontsize=9, pad=4)
ax2.legend(facecolor='#222', labelcolor='white', edgecolor='#444', fontsize=8)

plt.tight_layout()
path = os.path.join(OUT_DIR, "uncertainty_profile.png")
plt.savefig(path, dpi=130, bbox_inches='tight', facecolor=DARK)
plt.close(); print(f"  -> {path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3: 2D uncertainty volume heatmap (inline × pixel position)
# ─────────────────────────────────────────────────────────────────────────────
print("── Plotting 2D uncertainty volume heatmap ──")

# Collect row-mean uncertainty per inline (row = crossline direction after resize)
unc_map_rows = []
for idx in tqdm(range(TRIM, N_INLINES - TRIM), desc="Building uncertainty map"):
    if idx in panel_data:
        std_map = panel_data[idx]["std"]           # already computed
    else:
        img_t = load_inline(idx)
        _, std_map, _ = mc_uncertainty(img_t, n_mc=8, t_val=EXTRACT_T)  # fast pass
    unc_map_rows.append(std_map.mean(axis=1))      # collapse crossline axis -> (IMG_SIZE,)

unc_map = np.array(unc_map_rows).T     # (IMG_SIZE, N_inlines_trimmed)
np.save(os.path.join(OUT_DIR, "uncertainty_all.npy"), unc_map)

fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=DARK)
fig.suptitle(
    "2D Uncertainty Volume — Sleipner Seismic\n"
    "Row = crossline position (after 256×256 resize)  |  Column = inline (lateral position)\n"
    "Left: linear scale  |  Right: log scale (reveals low-uncertainty structure)",
    color='white', fontsize=11, y=1.02
)

for ax, norm, title, cmap in [
    (axes[0], None,                  "Pixel Uncertainty σ (linear)", "hot"),
    (axes[1], LogNorm(vmin=max(unc_map.min(), 1e-5), vmax=unc_map.max()),
                                     "Pixel Uncertainty σ (log scale)", "inferno"),
]:
    ax.set_facecolor(PANEL)
    im = ax.imshow(unc_map, aspect='auto', cmap=cmap, norm=norm,
                   extent=[TRIM, N_INLINES-TRIM, IMG_SIZE, 0],
                   interpolation='nearest')
    # Mark panel inlines
    for idx in PANEL_INLINES:
        ax.axvline(idx, color='#a5d6a7', lw=0.8, alpha=0.7, ls='--')
    ax.set_xlabel('Inline index (lateral position)', color='#aaa', fontsize=9)
    ax.set_ylabel('Crossline position (resized)', color='#aaa', fontsize=9)
    ax.set_title(title, color='white', fontsize=9, pad=4)
    ax.tick_params(colors='#aaa', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#333')
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label('σ (pixel uncertainty)', color='#ccc', fontsize=8)
    cb.ax.tick_params(colors='#aaa', labelsize=7)

plt.tight_layout()
path = os.path.join(OUT_DIR, "uncertainty_volume.png")
plt.savefig(path, dpi=140, bbox_inches='tight', facecolor=DARK)
plt.close(); print(f"  -> {path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4: Uncertainty distribution + insights summary
# ─────────────────────────────────────────────────────────────────────────────
print("── Plotting uncertainty histogram & insights ──")

all_stds_flat = np.concatenate([panel_data[i]["std"].ravel() for i in PANEL_INLINES])
all_cvs_flat  = np.clip(np.concatenate([panel_data[i]["cv"].ravel() for i in PANEL_INLINES]), 0, 2)

fig = plt.figure(figsize=(18, 10), facecolor=DARK)
gs  = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35,
                        left=0.08, right=0.97, top=0.90, bottom=0.08)
fig.suptitle(
    "Uncertainty Distribution & Key Insights  —  PDRM Monte Carlo Analysis\n"
    f"Sleipner 4D Seismic  |  N={N_MC} MC passes  |  t={EXTRACT_T}",
    color='white', fontsize=11, y=0.97,
)

def styled_ax(ax, title='', xl='', yl=''):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors='#aaa', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#333')
    if title: ax.set_title(title, color='white', fontsize=9, pad=4)
    if xl:    ax.set_xlabel(xl, color='#aaa', fontsize=8)
    if yl:    ax.set_ylabel(yl, color='#aaa', fontsize=8)

# (0,0) Histogram of pixel std
ax = fig.add_subplot(gs[0, 0])
styled_ax(ax, "Pixel Uncertainty Distribution (σ)", "σ (std across MC passes)", "Pixel count")
ax.hist(all_stds_flat, bins=80, color='#4fc3f7', alpha=0.85, edgecolor='none')
p50 = np.percentile(all_stds_flat, 50)
p90 = np.percentile(all_stds_flat, 90)
ax.axvline(p50, color='#ffcc80', lw=1.5, ls='--', label=f'Median σ={p50:.4f}')
ax.axvline(p90, color='#ff7043', lw=1.5, ls='--', label=f'90th pct σ={p90:.4f}')
ax.legend(facecolor='#222', labelcolor='white', edgecolor='#444', fontsize=7)

# (0,1) Coefficient of variation histogram
ax = fig.add_subplot(gs[0, 1])
styled_ax(ax, "Coefficient of Variation (σ/|μ|)", "CV", "Pixel count")
ax.hist(all_cvs_flat[all_cvs_flat < 1.5], bins=80, color='#a5d6a7', alpha=0.85, edgecolor='none')
ax.axvline(0.3, color='#ff7043', lw=1.5, ls='--', label='High uncertainty (CV>0.3)')
ax.legend(facecolor='#222', labelcolor='white', edgecolor='#444', fontsize=7)

# (0,2) Uncertainty per inline (mean sigma)
ax = fig.add_subplot(gs[0, 2])
styled_ax(ax, "Mean Uncertainty by Inline (all 249)", "Inline index", "Mean σ")
ax.plot(inline_ids, vol_mean_unc, color='#4fc3f7', lw=1.2, alpha=0.9)
ax.fill_between(inline_ids, vol_mean_unc,
                where=vol_mean_unc > np.percentile(vol_mean_unc, 80),
                color='#ff7043', alpha=0.4, label='High-uncertainty inlines (>80th pct)')
for idx in PANEL_INLINES:
    ax.scatter(idx, vol_mean_unc[idx], color='#a5d6a7', s=30, zorder=5)
ax.legend(facecolor='#222', labelcolor='white', edgecolor='#444', fontsize=7)

# (1,0) Scatter: original amplitude variance vs uncertainty
ax = fig.add_subplot(gs[1, 0])
styled_ax(ax, "Amplitude Heterogeneity vs Uncertainty", "Inline amplitude std", "Mean pixel σ")
amp_stds = [vol[i].std() for i in range(N_INLINES)]
ax.scatter(amp_stds, vol_mean_unc, c=inline_ids, cmap='Blues', s=10, alpha=0.7)
# Correlation
corr = np.corrcoef(amp_stds, vol_mean_unc)[0, 1]
ax.text(0.05, 0.92, f"Pearson r = {corr:.3f}", transform=ax.transAxes,
        color='#ffcc80', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', fc='#111', ec='#555', alpha=0.85))

# (1,1) Per-inline: mean uncertainty vs max uncertainty scatter
ax = fig.add_subplot(gs[1, 1])
styled_ax(ax, "Mean vs Peak Uncertainty by Inline", "Mean σ", "Max σ")
sc = ax.scatter(vol_mean_unc, vol_max_unc, c=inline_ids, cmap='Blues', s=10, alpha=0.75)
# Highlight panel inlines
for idx in PANEL_INLINES:
    ax.annotate(str(idx), (vol_mean_unc[idx], vol_max_unc[idx]),
                fontsize=7, color='#a5d6a7',
                xytext=(4, 4), textcoords='offset points')
cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
cb.set_label('Inline index', color='#ccc', fontsize=7)
cb.ax.tick_params(colors='#aaa', labelsize=6)

# (1,2) Key findings text panel
ax = fig.add_subplot(gs[1, 2]); ax.axis('off'); ax.set_facecolor('#111')
high_unc_idx = np.where(vol_mean_unc > np.percentile(vol_mean_unc, 80))[0]
low_unc_idx  = np.where(vol_mean_unc < np.percentile(vol_mean_unc, 20))[0]
findings = [
    "KEY INSIGHTS",
    "",
    f"Global median uncertainty: {np.median(vol_mean_unc):.4f}",
    f"Global range: [{vol_mean_unc.min():.4f}, {vol_mean_unc.max():.4f}]",
    "",
    f"High-uncertainty inlines (>80th pct):",
    f"  {high_unc_idx[:8].tolist()}...",
    f"Low-uncertainty inlines (<20th pct):",
    f"  {low_unc_idx[:8].tolist()}...",
    "",
    f"Amplitude-uncertainty correlation: r={corr:.3f}",
    f"  {'Strong' if abs(corr)>0.6 else 'Moderate' if abs(corr)>0.3 else 'Weak'} link between",
    f"  seismic heterogeneity and model",
    f"  uncertainty.",
    "",
    "Interpretation guidance:",
    "  High σ → structurally complex zones",
    "  where model sees ambiguous seismic",
    "  character. NOT necessarily geology.",
    "  Low σ  → homogeneous zones where",
    "  model is consistently confident.",
    "",
    "CCS relevance: anomalous uncertainty",
    "  spikes may indicate zones with",
    "  seismic character outside the model",
    "  training distribution (e.g. CO2",
    "  plume edges or acquisition effects).",
]
ax.text(0.05, 0.97, '\n'.join(findings), transform=ax.transAxes,
        color='#ccc', fontsize=7.5, va='top', ha='left', family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', fc='#111', ec='#444', alpha=0.95))

plt.tight_layout()
path = os.path.join(OUT_DIR, "uncertainty_histogram.png")
plt.savefig(path, dpi=130, bbox_inches='tight', facecolor=DARK)
plt.close(); print(f"  -> {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Uncertainty analysis complete.")
print(f"  Global mean sigma : {vol_mean_unc.mean():.4f}")
print(f"  High-unc inlines  : {high_unc_idx[:10].tolist()}")
print(f"  Low-unc  inlines  : {low_unc_idx[:10].tolist()}")
print(f"  Amplitude-unc r   : {corr:.3f}")
print(f"\n  Outputs:")
for f in ["uncertainty_panel.png", "uncertainty_profile.png",
          "uncertainty_volume.png", "uncertainty_histogram.png",
          "uncertainty_profile.npy", "uncertainty_all.npy"]:
    p = os.path.join(OUT_DIR, f)
    if os.path.exists(p):
        print(f"    {p}")
print(f"{'='*60}")
