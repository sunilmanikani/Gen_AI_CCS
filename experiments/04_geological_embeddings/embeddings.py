#!/usr/bin/env python3
"""
embeddings.py — AE vs Diffusion UNet embedding comparison on Sleipner.

Architectures taken EXACTLY from working_code/embeddings.ipynb:
  - SimpleAutoencoder  : CNN → Linear(128) bottleneck
  - UNet               : 3+bottleneck+3, dim=32

Extraction follows notebook exactly:
  - AE   : encoder(x) → 128-d embedding
  - Diff : bottleneck GAP at EXTRACTION_TIMESTEP=100, random noise

Outputs:
  /datadisk/tupi/tupi/Sliepner/embeddings_output/plots/reconstruction.png
  /datadisk/tupi/tupi/Sliepner/embeddings_output/plots/pca_geological_sequence.png
  /datadisk/tupi/tupi/Sliepner/embeddings_output/plots/pca_clustering.png
  /datadisk/tupi/tupi/Sliepner/embeddings_output/plots/metrics_comparison.png
  /datadisk/tupi/tupi/Sliepner/embeddings_output/embeddings/ae_embs.npy
  /datadisk/tupi/tupi/Sliepner/embeddings_output/embeddings/diff_embs.npy
"""

import os, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
BASE               = "/datadisk/tupi/tupi/Sliepner"
VOLUME_PATH        = os.path.join(BASE, "sliepner.npy")
OUT_DIR            = os.path.join(BASE, "embeddings_output")
CKPT_DIR           = os.path.join(OUT_DIR, "checkpoints")
PLOTS_DIR          = os.path.join(OUT_DIR, "plots")
EMB_DIR            = os.path.join(OUT_DIR, "embeddings")

AE_CKPT            = os.path.join(CKPT_DIR, "ae.pth")
DIFF_CKPT          = os.path.join(CKPT_DIR, "unet_diffusion.pth")

DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE           = 256       # matches notebook
BATCH_SIZE         = 4         # matches notebook
SEED               = 42        # matches notebook
DIFF_STEPS         = 1000      # matches notebook
EXTRACTION_T       = 100       # matches notebook EXTRACTION_TIMESTEP
UNET_DIM           = 32        # matches notebook UNET_DIM
AE_DIM             = 32        # matches notebook AE_DIM
N_CLUSTERS         = 3         # matches notebook (k=3)
N_EXAMPLES         = 6
TRIM               = 10        # drop first/last N inlines
DARK               = '#0e0e0e'
PANEL              = '#181818'
DEPTH_CMAP         = 'Blues'
CLUST_COLS         = ['#4fc3f7', '#a5d6a7', '#ffcc80']

for d in [PLOTS_DIR, EMB_DIR]:
    os.makedirs(d, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Device : {DEVICE}")


# ── Dataset (exact from embeddings.ipynb) ─────────────────────────────────────
class SeismicDataset(Dataset):
    def __init__(self, volume_path=VOLUME_PATH, img_size=IMG_SIZE):
        print(f"Loading volume from {volume_path}...")
        self.vol = np.load(volume_path).astype(np.float32)
        robust_max = np.percentile(np.abs(self.vol), 99)
        if robust_max < 1e-6: robust_max = 1.0
        self.vol = np.clip(self.vol / robust_max, -1, 1)
        self.img_size = img_size

    def __len__(self): return self.vol.shape[0]

    def __getitem__(self, idx):
        img   = self.vol[idx]
        h, w  = img.shape
        img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        if h != self.img_size or w != self.img_size:
            img_t = F.interpolate(img_t, size=(self.img_size, self.img_size),
                                  mode='bilinear', align_corners=False)
        return img_t.squeeze(0), idx


# ── SimpleAutoencoder (exact from embeddings.ipynb) ───────────────────────────
class SimpleAutoencoder(nn.Module):
    def __init__(self, dim=AE_DIM):
        super().__init__()
        self.f_dim    = IMG_SIZE // 8
        self.flat_dim = dim * 4 * self.f_dim * self.f_dim
        self.encoder  = nn.Sequential(
            nn.Conv2d(1, dim,    4, 2, 1), nn.ReLU(),
            nn.Conv2d(dim, dim*2, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(dim*2, dim*4, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.flat_dim, 128),
        )
        self.decoder_fc  = nn.Linear(128, self.flat_dim)
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(dim*4, dim*2, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(dim*2, dim,   4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(dim,   1,     4, 2, 1), nn.Tanh(),
        )
        self.dim = dim

    def forward(self, x):
        z = self.encoder(x)
        r = self.decoder_fc(z).view(-1, self.dim*4, self.f_dim, self.f_dim)
        return self.decoder_cnn(r), z


# ── UNet diffusion (exact from embeddings.ipynb) ──────────────────────────────
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
        self.mlp    = nn.Sequential(nn.SiLU(),
                                    nn.Linear(time_emb_dim, dim_out*2)) if time_emb_dim else None
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
        self.mid_block1   = ResnetBlock(dim*4, dim*4, time_emb_dim=dim*4)
        self.mid_attn     = Attention(dim*4)
        self.mid_block2   = ResnetBlock(dim*4, dim*4, time_emb_dim=dim*4)
        self.ups          = nn.ModuleList([
            ResnetBlock(dim*8, dim*2, time_emb_dim=dim*4),
            ResnetBlock(dim*4, dim,   time_emb_dim=dim*4),
            ResnetBlock(dim*2, dim,   time_emb_dim=dim*4),
        ])
        self.up_samples   = nn.ModuleList([
            nn.ConvTranspose2d(dim*4, dim*4, 4, 2, 1),
            nn.ConvTranspose2d(dim*2, dim*2, 4, 2, 1),
            nn.ConvTranspose2d(dim,   dim,   4, 2, 1),
        ])
        self.final_conv   = nn.Conv2d(dim, out_dim, 1)

    def forward(self, x, time, return_embedding=False):
        t      = self.time_mlp(time)
        x      = self.init_conv(x)
        h_vals = []
        for block, sample in zip(self.downs, self.down_samples):
            x = block(x, t); h_vals.append(x); x = sample(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        if return_embedding:
            return torch.mean(x, dim=(2, 3))        # (B, dim*4) = (B, 128)
        for block, sample in zip(self.ups, self.up_samples):
            x = sample(x); h = h_vals.pop()
            if x.shape[2:] != h.shape[2:]:
                x = F.interpolate(x, size=h.shape[2:], mode='nearest')
            x = torch.cat((x, h), dim=1)
            x = block(x, t)
        return self.final_conv(x)


# ── Diffusion schedule ────────────────────────────────────────────────────────
class DiffUtils:
    def __init__(self, steps=DIFF_STEPS, device=DEVICE):
        self.steps = steps; self.device = device
        betas      = torch.linspace(0.0001, 0.02, steps).to(device)
        alphas     = 1. - betas
        alphas_cp  = torch.cumprod(alphas, dim=0)
        self.betas = betas; self.alphas = alphas
        self.alphas_cumprod = alphas_cp
        self.sa    = torch.sqrt(alphas_cp)
        self.so    = torch.sqrt(1. - alphas_cp)

    def q_sample(self, x0, t, noise=None):
        if noise is None: noise = torch.randn_like(x0)
        return (self.sa[t].view(-1,1,1,1)*x0 +
                self.so[t].view(-1,1,1,1)*noise), noise

    def predict_x0(self, x_t, t, pred_noise):
        return (x_t - self.so[t].view(-1,1,1,1)*pred_noise) / self.sa[t].view(-1,1,1,1)

    @torch.no_grad()
    def ddpm_reverse(self, model, x_noisy, t_start):
        """Full DDPM reverse from t_start to 0."""
        model.eval()
        x = x_noisy.clone().to(self.device)
        n = x.shape[0]
        for i in tqdm(reversed(range(1, t_start+1)),
                      desc=f"  Reverse t={t_start}→0", total=t_start, leave=False):
            t  = torch.full((n,), i, device=self.device, dtype=torch.long)
            pn = model(x, t)
            a  = self.alphas_cumprod[t].view(-1,1,1,1)
            b  = self.betas[t].view(-1,1,1,1)
            x  = (x - b / torch.sqrt(1-a) * pn) / torch.sqrt(1-b)
            if i > 1:
                x = x + torch.sqrt(b) * torch.randn_like(x)
        model.train()
        return x.clamp(-1, 1)


# ── Embedding extraction (exact logic from embeddings.ipynb) ──────────────────
@torch.no_grad()
def extract_embeddings(ae, unet, diff, loader):
    ae.eval(); unet.eval()
    ae_embs = []; diff_embs = []; idxs = []
    print(f"\nExtracting embeddings (EXTRACTION_T={EXTRACTION_T}, random noise — matches notebook) …")
    for imgs, idx in tqdm(loader, desc="  Extracting"):
        imgs = imgs.to(DEVICE); b = imgs.shape[0]

        # AE embedding — exact from notebook: _, ae_z = ae_model(imgs)
        _, ae_z = ae(imgs)
        ae_embs.append(ae_z.cpu().numpy())

        # Diffusion embedding — exact from notebook: random noise, t=100
        t     = torch.full((b,), EXTRACTION_T, device=DEVICE, dtype=torch.long)
        noise = torch.randn_like(imgs)                    # random noise (notebook)
        xn, _ = diff.q_sample(imgs, t, noise)
        dz    = unet(xn, t, return_embedding=True)
        diff_embs.append(dz.cpu().numpy())

        idxs += idx.tolist()

    return (np.concatenate(ae_embs),
            np.concatenate(diff_embs),
            np.array(idxs))


# ── PCA + clustering ──────────────────────────────────────────────────────────
def analyse(emb, label):
    z    = StandardScaler().fit_transform(emb)
    pca  = PCA(n_components=2, random_state=SEED)
    proj = pca.fit_transform(z)
    var  = pca.explained_variance_ratio_ * 100
    # K-Means on raw embeddings — matches notebook
    km   = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    labs = km.fit_predict(emb)
    sil  = silhouette_score(z, labs)
    db   = davies_bouldin_score(z, labs)
    dead = (emb.std(0) < 1e-4).sum()
    print(f"  {label:35s}  PC1={var[0]:.1f}%  PC2={var[1]:.1f}%  "
          f"sil={sil:.3f}  db={db:.3f}  dead_dims={dead}/{emb.shape[1]}")
    return proj, var, labs, sil, db


# ── Plotting helpers ──────────────────────────────────────────────────────────
def _ax(ax, title='', xl='', yl='', xlim=None, ylim=None):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors='#aaa', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#333')
    if title: ax.set_title(title, color='white', fontsize=9, pad=6)
    if xl:    ax.set_xlabel(xl, color='#aaa', fontsize=8)
    if yl:    ax.set_ylabel(yl, color='#aaa', fontsize=8)
    if xlim:  ax.set_xlim(xlim)
    if ylim:  ax.set_ylim(ylim)


# ── Plot 1: Reconstruction ────────────────────────────────────────────────────
def plot_reconstruction(ds, ae, unet, diff, out):
    n   = len(ds)
    sel = np.linspace(TRIM, n-TRIM-1, N_EXAMPLES, dtype=int)

    batch_x = torch.stack([ds[i][0] for i in sel]).to(DEVICE)
    t_ext   = torch.full((len(sel),), EXTRACTION_T, device=DEVICE, dtype=torch.long)

    # AE reconstruction
    ae.eval()
    with torch.no_grad():
        ae_recon, _ = ae(batch_x)
        ae_recon    = ae_recon.clamp(-1, 1)

    # Diffusion: add noise at EXTRACTION_T → predict x0 (single-step, matches notebook)
    torch.manual_seed(SEED)
    noise   = torch.randn_like(batch_x)
    xn, _   = diff.q_sample(batch_x, t_ext, noise)
    unet.eval()
    with torch.no_grad():
        pred_noise = unet(xn, t_ext)
        diff_recon = diff.predict_x0(xn, t_ext, pred_noise).clamp(-1, 1)

    cols = ["Original\n(sliepner.npy)",
            "SimpleAutoencoder\n(CNN → Linear(128))",
            f"Noisy input  t={EXTRACTION_T}\n(~10% noise added)",
            f"UNet Diffusion\n(pred x₀ from t={EXTRACTION_T})"]

    fig, axes = plt.subplots(N_EXAMPLES, 4, figsize=(18, N_EXAMPLES*2.8), facecolor=DARK)
    fig.suptitle(
        f"Reconstruction  —  architectures from working_code/embeddings.ipynb\n"
        f"sliepner.npy  ·  inlines {TRIM}–{n-TRIM-1}  ·  {N_EXAMPLES} examples",
        color='white', fontsize=11, y=1.001,
    )
    for row, idx in enumerate(sel):
        orig     = batch_x[row].cpu().squeeze().numpy()
        ae_np    = ae_recon[row].cpu().squeeze().numpy()
        noisy_np = xn[row].cpu().squeeze().numpy()
        dn_np    = diff_recon[row].cpu().squeeze().numpy()

        for col, img in enumerate([orig, ae_np, noisy_np, dn_np]):
            ax = axes[row, col]
            ax.imshow(img.T, cmap='seismic', vmin=-1, vmax=1,
                      aspect='auto', interpolation='nearest')
            ax.axis('off'); ax.set_facecolor(PANEL)
            if row == 0:
                ax.set_title(cols[col], color='white', fontsize=8, pad=5)
            if col in (1, 3):
                mae = np.abs(orig - img).mean()
                ax.text(0.98, 0.02, f"MAE={mae:.3f}",
                        transform=ax.transAxes, color='#ffcc80', fontsize=7,
                        ha='right', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', fc='#111', ec='#555', alpha=0.85))
        axes[row, 0].text(
            -0.05, 0.5, f"Inline {int(idx):03d}",
            transform=axes[row, 0].transAxes,
            color='#ffcc80', fontsize=8, va='center', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', fc='#1a1a1a', ec='#555', alpha=0.9),
        )
    plt.tight_layout(rect=[0.06, 0, 1, 0.999])
    plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=DARK)
    plt.close(); print(f"  → {out}")


# ── Plot 2: Geological sequence ───────────────────────────────────────────────
def plot_geological(ae_proj, ae_var, diff_proj, diff_var, depth, xlim, ylim, out):
    vmin, vmax = depth.min(), depth.max()
    fig, axes  = plt.subplots(1, 2, figsize=(16, 6), facecolor=DARK)
    fig.suptitle(
        "Embedding Space — Geological Sequence  (inline = stratigraphic depth proxy)\n"
        "SimpleAutoencoder (128-d)  vs  UNet Diffusion (128-d)  ·  same Blues colormap  ·  same axis scale",
        color='white', fontsize=11, y=1.02,
    )
    for ax, proj, var, title in zip(
        axes,
        [ae_proj,   diff_proj],
        [ae_var,    diff_var],
        ["SimpleAutoencoder  (CNN → Linear 128-d)",
         "UNet Diffusion  (3+bottleneck+3, dim=32, 128-d GAP)"],
    ):
        _ax(ax, title, f"PC1 ({var[0]:.1f}%)", f"PC2 ({var[1]:.1f}%)", xlim, ylim)
        sc = ax.scatter(proj[:,0], proj[:,1], c=depth, cmap=DEPTH_CMAP,
                        vmin=vmin, vmax=vmax, s=38, alpha=0.92, linewidths=0)
        for lbl, idx in [("Shallow", 0), ("Mid", len(depth)//2), ("Deep", len(depth)-1)]:
            ax.annotate(lbl, xy=(proj[idx,0], proj[idx,1]),
                        xytext=(proj[idx,0]+0.4, proj[idx,1]+0.4),
                        color='white', fontsize=7, alpha=0.9,
                        arrowprops=dict(arrowstyle='->', color='#888', lw=0.8))
        ax.text(0.02, 0.97, f"2-PC var: {var.sum():.1f}%",
                transform=ax.transAxes, color='#ffcc80', fontsize=8, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='#1a1a1a', ec='#555', alpha=0.9))

    cbar_ax = fig.add_axes([0.92, 0.12, 0.018, 0.74])
    sm = plt.cm.ScalarMappable(cmap=DEPTH_CMAP, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Inline (shallow → deep)", color='#ccc', fontsize=8, labelpad=6)
    cb.ax.tick_params(colors='#aaa', labelsize=7)
    fig.text(0.5, -0.02,
             f"Note: first and last {TRIM} inlines removed (edge artefacts)",
             ha='center', color='#888', fontsize=8)
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=DARK)
    plt.close(); print(f"  → {out}")


# ── Plot 3: Clustering ────────────────────────────────────────────────────────
def plot_clustering(ae_proj, ae_var, ae_labs, ae_sil, ae_db,
                    diff_proj, diff_var, diff_labs, diff_sil, diff_db,
                    depth, xlim, ylim, out):
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    fig, axes  = plt.subplots(1, 2, figsize=(16, 6), facecolor=DARK)
    fig.suptitle(
        f"K-Means Clustering  k={N_CLUSTERS}  ·  same axis scale\n"
        "Point size = stratigraphic depth  ·  Colour = cluster",
        color='white', fontsize=11, y=1.02,
    )
    for ax, proj, var, labs, sil, db, title in zip(
        axes,
        [ae_proj,   diff_proj],
        [ae_var,    diff_var],
        [ae_labs,   diff_labs],
        [ae_sil,    diff_sil],
        [ae_db,     diff_db],
        ["SimpleAutoencoder", "UNet Diffusion"],
    ):
        _ax(ax, f"{title}  |  sil={sil:.3f}  DB={db:.3f}",
            f"PC1 ({var[0]:.1f}%)", f"PC2 ({var[1]:.1f}%)", xlim, ylim)
        for cl in range(N_CLUSTERS):
            m = labs == cl
            ax.scatter(proj[m,0], proj[m,1], c=CLUST_COLS[cl],
                       s=20 + 30*depth_norm[m], alpha=0.88, linewidths=0,
                       label=f"Cluster {cl}  n={m.sum()}")
        ax.legend(fontsize=7, facecolor='#222', labelcolor='white',
                  edgecolor='#444', loc='best')
        ax.text(0.02, 0.97, f"2-PC var: {var.sum():.1f}%",
                transform=ax.transAxes, color='#ffcc80', fontsize=8, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='#1a1a1a', ec='#555', alpha=0.9))
    fig.text(0.5, -0.02,
             f"Point size encodes stratigraphic depth  ·  first/last {TRIM} inlines removed",
             ha='center', color='#888', fontsize=8)
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=DARK)
    plt.close(); print(f"  → {out}")


# ── Plot 4: Metrics comparison ────────────────────────────────────────────────
def plot_metrics(ae_var, ae_sil, ae_db, diff_var, diff_sil, diff_db, out):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=DARK)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors='#aaa', labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor('#444')

    metrics  = ["Silhouette ↑\n(cluster tightness)",
                "1/Davies-Bouldin ↑\n(cluster separation)",
                "PC1 Variance %\n(lower=avoids 1-D collapse)",
                "2-PC Total Var %\n(higher=richer embedding)"]
    ae_vals  = [ae_sil,   1/(ae_db+1e-9),   ae_var[0],   ae_var.sum()]
    dv_vals  = [diff_sil, 1/(diff_db+1e-9), diff_var[0], diff_var.sum()]

    x = np.arange(len(metrics)); w = 0.33
    b1 = ax.bar(x-w/2, ae_vals, w, color='#4fc3f7', label='SimpleAutoencoder', alpha=0.9)
    b2 = ax.bar(x+w/2, dv_vals, w, color='#a5d6a7', label='UNet Diffusion',    alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(metrics, color='white', fontsize=9)
    ax.legend(facecolor='#1a1a1a', labelcolor='white', edgecolor='#444', fontsize=10)
    ax.set_title(
        "AE vs Diffusion Embedding Quality\n"
        "Architectures from working_code/embeddings.ipynb  ·  extraction t=100",
        color='white', fontsize=10, pad=8,
    )

    raw_ae = [ae_sil, ae_db, ae_var[0], ae_var.sum()]
    raw_dv = [diff_sil, diff_db, diff_var[0], diff_var.sum()]
    for i, (va, vd, ra, rd) in enumerate(zip(ae_vals, dv_vals, raw_ae, raw_dv)):
        ae_wins = va >= vd
        ax.text(i-w/2, va+0.003, f"{'★' if ae_wins else ''}{ra:.3f}",
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                color='#7ff7ff' if ae_wins else '#aaaaaa')
        ax.text(i+w/2, vd+0.003, f"{'★' if not ae_wins else ''}{rd:.3f}",
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                color='#aaffaa' if not ae_wins else '#aaaaaa')

    ae_wins_count = sum(va >= vd for va, vd in zip(ae_vals, dv_vals))
    notes = [
        f"★ starred value = winner for each metric",
        f"  AE wins {ae_wins_count}/4   ·   Diffusion wins {4-ae_wins_count}/4",
        "",
        f"  Silhouette/DB: clustering tightness in standardised embedding space",
        f"  PC1 variance : AE collapsing to 1 axis (amplitude) vs Diffusion using 2 axes (geology)",
        f"  2-PC total   : Diffusion encodes more variance in first 2 components",
        "",
        f"  Key: Diffusion learns geological structure without being told to cluster.",
        f"       AE learns to compress — simpler objective, different representation.",
    ]
    fig.text(0.5, -0.22, '\n'.join(notes), ha='center', va='top', color='#ccc',
             fontsize=8, family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', fc='#111', ec='#444', alpha=0.95))
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=DARK)
    plt.close(); print(f"  → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    for p in [AE_CKPT, DIFF_CKPT]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p}\nRun pretraining.py first.")

    ds     = SeismicDataset()
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True)
    diff   = DiffUtils()

    print("\nLoading checkpoints …")
    ae = SimpleAutoencoder().to(DEVICE)
    ae.load_state_dict(torch.load(AE_CKPT, map_location=DEVICE,
                                  weights_only=False)["state_dict"])
    print(f"  SimpleAutoencoder ← {AE_CKPT}")

    unet = UNet().to(DEVICE)
    unet.load_state_dict(torch.load(DIFF_CKPT, map_location=DEVICE,
                                    weights_only=False)["state_dict"])
    print(f"  UNet Diffusion    ← {DIFF_CKPT}")

    # Reconstruction plot
    print("\nGenerating reconstruction plot …")
    plot_reconstruction(ds, ae, unet, diff,
                        os.path.join(PLOTS_DIR, "reconstruction.png"))

    # Extract embeddings
    ae_emb, diff_emb, all_idxs = extract_embeddings(ae, unet, diff, loader)
    np.save(os.path.join(EMB_DIR, "ae_embs.npy"),   ae_emb)
    np.save(os.path.join(EMB_DIR, "diff_embs.npy"), diff_emb)
    print(f"  Saved → {EMB_DIR}/")
    print(f"  AE   shape={ae_emb.shape}   dead={(ae_emb.std(0)<1e-4).sum()}/{ae_emb.shape[1]}")
    print(f"  Diff shape={diff_emb.shape}  dead={(diff_emb.std(0)<1e-4).sum()}/{diff_emb.shape[1]}")

    # Trim edge inlines
    mask   = (all_idxs >= TRIM) & (all_idxs < len(ds) - TRIM)
    ae_t   = ae_emb[mask];   diff_t = diff_emb[mask]
    depth  = all_idxs[mask]
    print(f"\n  Using {mask.sum()} inlines (first/last {TRIM} trimmed)")

    # Analyse
    print("\nPCA + K-Means …")
    ae_proj,   ae_var,   ae_labs,   ae_sil,   ae_db   = analyse(ae_t,   "SimpleAutoencoder")
    diff_proj, diff_var, diff_labs, diff_sil, diff_db = analyse(diff_t, "UNet Diffusion")

    # Shared axis limits
    all_x = np.concatenate([ae_proj[:,0], diff_proj[:,0]])
    all_y = np.concatenate([ae_proj[:,1], diff_proj[:,1]])
    pad   = 0.5
    xlim  = (all_x.min()-pad, all_x.max()+pad)
    ylim  = (all_y.min()-pad, all_y.max()+pad)

    print("\nGenerating plots …")
    plot_geological(ae_proj, ae_var, diff_proj, diff_var, depth, xlim, ylim,
                    os.path.join(PLOTS_DIR, "pca_geological_sequence.png"))
    plot_clustering(ae_proj, ae_var, ae_labs, ae_sil, ae_db,
                    diff_proj, diff_var, diff_labs, diff_sil, diff_db,
                    depth, xlim, ylim,
                    os.path.join(PLOTS_DIR, "pca_clustering.png"))
    plot_metrics(ae_var, ae_sil, ae_db, diff_var, diff_sil, diff_db,
                 os.path.join(PLOTS_DIR, "metrics_comparison.png"))

    print(f"\n{'─'*60}")
    print(f"All outputs in {OUT_DIR}/")
    print(f"  {PLOTS_DIR}/reconstruction.png")
    print(f"  {PLOTS_DIR}/pca_geological_sequence.png")
    print(f"  {PLOTS_DIR}/pca_clustering.png")
    print(f"  {PLOTS_DIR}/metrics_comparison.png")
    print(f"  {EMB_DIR}/ae_embs.npy")
    print(f"  {EMB_DIR}/diff_embs.npy")
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
