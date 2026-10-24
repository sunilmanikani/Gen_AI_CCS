#!/usr/bin/env python3
"""
uncertainty_overlay.py
4 selected slices: original | seismic + uncertainty overlay | uncertainty only
Output: /datadisk/tupi/tupi/Sliepner/uncertainty_output/uncertainty_overlay.png
"""
import os, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

BASE      = "/datadisk/tupi/tupi/Sliepner"
DIFF_CKPT = os.path.join(BASE, "embeddings_output/checkpoints/unet_diffusion.pth")
VOL_PATH  = os.path.join(BASE, "sliepner.npy")
OUT       = os.path.join(BASE, "uncertainty_output/uncertainty_overlay.png")

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE  = 256
N_MC      = 30
T_VAL     = 100
SEED      = 42
DARK      = '#0e0e0e'
PANEL     = '#181818'

# 4 slices: low-unc, peak-unc cluster, mid-volume, late-volume
SLICES = [20, 55, 124, 180]

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── Architecture ──────────────────────────────────────────────────────────────
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
    def __init__(self, dim=32, out_dim=1, channels=1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim), nn.Linear(dim, dim*4),
            nn.GELU(), nn.Linear(dim*4, dim*4))
        self.init_conv    = nn.Conv2d(channels, dim, 7, padding=3)
        self.downs        = nn.ModuleList([
            ResnetBlock(dim,   dim,   time_emb_dim=dim*4),
            ResnetBlock(dim,   dim*2, time_emb_dim=dim*4),
            ResnetBlock(dim*2, dim*4, time_emb_dim=dim*4)])
        self.down_samples = nn.ModuleList([
            nn.Conv2d(dim,   dim,   4, 2, 1),
            nn.Conv2d(dim*2, dim*2, 4, 2, 1),
            nn.Conv2d(dim*4, dim*4, 4, 2, 1)])
        self.mid_block1 = ResnetBlock(dim*4, dim*4, time_emb_dim=dim*4)
        self.mid_attn   = Attention(dim*4)
        self.mid_block2 = ResnetBlock(dim*4, dim*4, time_emb_dim=dim*4)
        self.ups        = nn.ModuleList([
            ResnetBlock(dim*8, dim*2, time_emb_dim=dim*4),
            ResnetBlock(dim*4, dim,   time_emb_dim=dim*4),
            ResnetBlock(dim*2, dim,   time_emb_dim=dim*4)])
        self.up_samples = nn.ModuleList([
            nn.ConvTranspose2d(dim*4, dim*4, 4, 2, 1),
            nn.ConvTranspose2d(dim*2, dim*2, 4, 2, 1),
            nn.ConvTranspose2d(dim,   dim,   4, 2, 1)])
        self.final_conv = nn.Conv2d(dim, out_dim, 1)

    def forward(self, x, time):
        t = self.time_mlp(time); x = self.init_conv(x); h_vals = []
        for block, sample in zip(self.downs, self.down_samples):
            x = block(x, t); h_vals.append(x); x = sample(x)
        x = self.mid_block1(x, t); x = self.mid_attn(x); x = self.mid_block2(x, t)
        for block, sample in zip(self.ups, self.up_samples):
            x = sample(x); h = h_vals.pop()
            if x.shape[2:] != h.shape[2:]:
                x = F.interpolate(x, size=h.shape[2:], mode='nearest')
            x = torch.cat((x, h), dim=1); x = block(x, t)
        return self.final_conv(x)

# ── Diffusion utils ───────────────────────────────────────────────────────────
betas  = torch.linspace(0.0001, 0.02, 1000).to(DEVICE)
acp    = torch.cumprod(1. - betas, dim=0)
sa     = torch.sqrt(acp)
so     = torch.sqrt(1. - acp)

def q_sample(x0, t, noise):
    return sa[t].view(-1,1,1,1)*x0 + so[t].view(-1,1,1,1)*noise

def pred_x0(x_t, t, eps):
    return (x_t - so[t].view(-1,1,1,1)*eps) / sa[t].view(-1,1,1,1)

# ── Load model & volume ───────────────────────────────────────────────────────
print("Loading model...")
unet = UNet().to(DEVICE)
unet.load_state_dict(
    torch.load(DIFF_CKPT, map_location=DEVICE, weights_only=False)["state_dict"])
unet.eval()

print("Loading volume...")
vol = np.load(VOL_PATH).astype(np.float32)
robust_max = np.percentile(np.abs(vol), 99)
vol = np.clip(vol / robust_max, -1, 1)

def get_inline(idx):
    img = torch.tensor(vol[idx], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return F.interpolate(img, size=(IMG_SIZE, IMG_SIZE),
                         mode='bilinear', align_corners=False).squeeze(0)

# ── MC uncertainty per slice ───────────────────────────────────────────────────
@torch.no_grad()
def mc_unc(img_t):
    x0    = img_t.unsqueeze(0).to(DEVICE)
    t_ten = torch.full((1,), T_VAL, device=DEVICE, dtype=torch.long)
    recons = []
    for _ in range(N_MC):
        noise = torch.randn_like(x0)
        xn    = q_sample(x0, t_ten, noise)
        eps   = unet(xn, t_ten)
        x0h   = pred_x0(xn, t_ten, eps).clamp(-1, 1)
        recons.append(x0h.squeeze().cpu().numpy())
    recons = np.stack(recons)           # (N_MC, H, W)
    return recons.mean(0), recons.std(0)

print(f"Running {N_MC} MC passes for {len(SLICES)} slices...")
results = {}
for idx in SLICES:
    print(f"  Inline {idx}...", end=" ", flush=True)
    img_t = get_inline(idx)
    mean_r, std_r = mc_unc(img_t)
    results[idx] = {
        "orig": img_t.squeeze().numpy(),
        "mean": mean_r,
        "std":  std_r,
    }
    print(f"mean_sigma={std_r.mean():.4f}  max_sigma={std_r.max():.4f}")

# ── Global std range for consistent colour scale ──────────────────────────────
all_std = np.concatenate([results[i]["std"].ravel() for i in SLICES])
vmin_s  = float(np.percentile(all_std, 1))
vmax_s  = float(np.percentile(all_std, 99))
print(f"Uncertainty colour range: [{vmin_s:.4f}, {vmax_s:.4f}]")

# ── Plot ──────────────────────────────────────────────────────────────────────
# Layout: 4 rows (slices) × 3 columns (original | overlay | uncertainty only)
fig, axes = plt.subplots(
    len(SLICES), 3,
    figsize=(17, len(SLICES) * 4.2),
    facecolor=DARK,
    gridspec_kw={"wspace": 0.05, "hspace": 0.06},
)
# Reserve left=0.11 for amplitude cbar, right=0.89 for uncertainty cbar
fig.subplots_adjust(left=0.11, right=0.88, top=0.94, bottom=0.04)

fig.suptitle(
    f"Monte Carlo Uncertainty Overlay  —  PDRM  |  N={N_MC} passes  |  t={T_VAL}\n"
    "Left: seismic amplitude  ·  Centre: seismic + uncertainty overlay  "
    "·  Right: uncertainty only",
    color="white", fontsize=12, y=0.98,
)

col_titles = [
    "Original Seismic\n(normalised amplitude)",
    "Seismic  +  Uncertainty Overlay\n(hot colourmap, α = 0.55)",
    "Uncertainty Map Only\n(pixel σ across MC passes)",
]

# Normalise function for the 'hot' overlay
norm_unc = mcolors.Normalize(vmin=vmin_s, vmax=vmax_s)
cmap_unc = plt.cm.hot

for row, idx in enumerate(SLICES):
    d    = results[idx]
    orig = d["orig"].T       # transpose: time on y-axis
    std  = d["std"].T

    # ── Col 0: original seismic ───────────────────────────────────────────────
    ax = axes[row, 0]
    ax.imshow(orig, cmap="seismic", vmin=-1, vmax=1,
              aspect="auto", interpolation="nearest")
    ax.set_facecolor(PANEL); ax.axis("off")
    if row == 0:
        ax.set_title(col_titles[0], color="white", fontsize=9, pad=8)
    ax.text(-0.015, 0.5, f"Inline {idx:03d}\nσ̄={d['std'].mean():.4f}",
            transform=ax.transAxes, color="#ffcc80", fontsize=8,
            va="center", ha="right", rotation=0,
            bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a1a", ec="#555", alpha=0.9))

    # ── Col 1: seismic + uncertainty overlay ──────────────────────────────────
    ax = axes[row, 1]
    ax.imshow(orig, cmap="seismic", vmin=-1, vmax=1,
              aspect="auto", interpolation="nearest")
    # Build RGBA array for uncertainty overlay
    unc_rgba = cmap_unc(norm_unc(std))          # (H, W, 4)
    # Alpha: scale with normalised uncertainty so low-unc areas are transparent
    alpha_channel = np.clip((std - vmin_s) / (vmax_s - vmin_s + 1e-9), 0, 1)
    unc_rgba[..., 3] = alpha_channel * 0.7      # max alpha = 0.70
    ax.imshow(unc_rgba, aspect="auto", interpolation="nearest")
    ax.set_facecolor(PANEL); ax.axis("off")
    if row == 0:
        ax.set_title(col_titles[1], color="white", fontsize=9, pad=8)
    # Annotate peak uncertainty location
    peak_y, peak_x = np.unravel_index(std.argmax(), std.shape)
    ax.plot(peak_x, peak_y, "x", color="#00e5ff", ms=8, mew=1.8)
    ax.text(peak_x + 4, peak_y - 4, f"peak σ={std.max():.4f}",
            color="#00e5ff", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="#0a0a0a", ec="#00e5ff", alpha=0.8))

    # ── Col 2: uncertainty map only ───────────────────────────────────────────
    ax = axes[row, 2]
    im = ax.imshow(std, cmap="hot", vmin=vmin_s, vmax=vmax_s,
                   aspect="auto", interpolation="nearest")
    ax.set_facecolor(PANEL); ax.axis("off")
    if row == 0:
        ax.set_title(col_titles[2], color="white", fontsize=9, pad=8)
    # Stats annotation
    p50 = float(np.percentile(std, 50))
    p95 = float(np.percentile(std, 95))
    hi_pct = float((std > (vmin_s + 0.6*(vmax_s-vmin_s))).mean() * 100)
    ax.text(0.03, 0.97,
            f"median σ = {p50:.4f}\n"
            f"95th pct = {p95:.4f}\n"
            f"high-unc pixels: {hi_pct:.1f}%",
            transform=ax.transAxes, color="white", fontsize=7.5, va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="#111", ec="#666", alpha=0.88))

# ── Colourbar: uncertainty (right edge, well clear of plots) ─────────────────
# fig spans 0.88→1.0 on the right; put bar at 0.895 with width 0.018
cbar_ax = fig.add_axes([0.895, 0.10, 0.018, 0.80])
sm = plt.cm.ScalarMappable(cmap=cmap_unc, norm=norm_unc)
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label("Pixel uncertainty  σ\n(std across MC passes)",
             color="#ccc", fontsize=8, labelpad=10)
cb.ax.tick_params(colors="#aaa", labelsize=7)
cb.ax.yaxis.set_label_position("right")
cb.ax.yaxis.tick_right()

# ── Colourbar: seismic amplitude (left edge, well clear of plots) ─────────────
# fig spans 0.0→0.11 on the left; put bar at 0.025 with width 0.018
cbar_ax2 = fig.add_axes([0.025, 0.10, 0.018, 0.80])
sm2 = plt.cm.ScalarMappable(
    cmap="seismic", norm=mcolors.Normalize(vmin=-1, vmax=1))
cb2 = fig.colorbar(sm2, cax=cbar_ax2)
cb2.set_label("Normalised amplitude",
              color="#ccc", fontsize=8, labelpad=10)
cb2.ax.tick_params(colors="#aaa", labelsize=7)
cb2.ax.yaxis.set_label_position("left")
cb2.ax.yaxis.tick_left()

plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"\nSaved -> {OUT}")
print(f"Size  : {os.path.getsize(OUT)/1024:.0f} KB")
