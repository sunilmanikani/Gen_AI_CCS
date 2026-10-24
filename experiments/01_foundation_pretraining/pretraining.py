#!/usr/bin/env python3
"""
pretraining.py — Train AE and Diffusion UNet on Sleipner inlines.

Architectures taken EXACTLY from working_code/embeddings.ipynb:
  - SimpleAutoencoder : 3 Conv encoder → Linear(128) bottleneck → deconv decoder
  - UNet              : 3 encoder + bottleneck (attention) + 3 decoder, dim=32

Outputs:
  /datadisk/tupi/tupi/Sliepner/embeddings_output/checkpoints/ae.pth
  /datadisk/tupi/tupi/Sliepner/embeddings_output/checkpoints/unet_diffusion.pth
  /datadisk/tupi/tupi/Sliepner/embeddings_output/plots/pretraining_loss.png
"""

import os, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Config (mirrors embeddings.ipynb CONFIG) ──────────────────────────────────
VOLUME_PATH = "sliepner.npy"
OUT_DIR     = "embeddings_output"
CKPT_DIR    = os.path.join(OUT_DIR, "checkpoints")
PLOTS_DIR   = os.path.join(OUT_DIR, "plots")

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE    = 256
BATCH_SIZE  = 4       # matches notebook BATCH_SIZE
SEED        = 42
LR          = 1e-3    # matches notebook LR
UNET_DIM    = 32      # matches notebook UNET_DIM
AE_DIM      = 32      # matches notebook AE_DIM
DIFF_STEPS  = 1000    # matches notebook DIFFUSION_STEPS

AE_EPOCHS   = 100     # notebook uses 25 — we run longer overnight for convergence
DIFF_EPOCHS = 300     # notebook uses 25 — we run longer overnight for convergence

for d in [CKPT_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Device : {DEVICE}")
print(f"Architectures from working_code/embeddings.ipynb (exact)")
print(f"  SimpleAutoencoder : CNN → Linear(128)")
print(f"  UNet diffusion    : dim={UNET_DIM}, 3+bottleneck+3")
print(f"Epochs : AE={AE_EPOCHS}  Diffusion={DIFF_EPOCHS}")


# ── Dataset (exact from embeddings.ipynb RealSeismicDataset) ─────────────────
class SeismicDataset(Dataset):
    def __init__(self, volume_path=VOLUME_PATH, img_size=IMG_SIZE):
        print(f"Loading volume from {volume_path}...")
        self.vol = np.load(volume_path).astype(np.float32)
        robust_max = np.percentile(np.abs(self.vol), 99)
        if robust_max < 1e-6: robust_max = 1.0
        self.vol = np.clip(self.vol / robust_max, -1, 1)
        self.img_size = img_size
        print(f"  shape={self.vol.shape}  normalised to [-1,1]")

    def __len__(self): return self.vol.shape[0]

    def __getitem__(self, idx):
        img   = self.vol[idx]                                   # (468, 1001)
        h, w  = img.shape
        img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        if h != self.img_size or w != self.img_size:
            img_t = F.interpolate(img_t, size=(self.img_size, self.img_size),
                                  mode='bilinear', align_corners=False)
        return img_t.squeeze(0)                                 # (1, 256, 256)


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
        z     = self.encoder(x)
        r     = self.decoder_fc(z).view(-1, self.dim*4, self.f_dim, self.f_dim)
        return self.decoder_cnn(r), z


# ── UNet diffusion (exact from embeddings.ipynb) ──────────────────────────────
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, x):
        device  = x.device; half_dim = self.dim // 2
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
                                    nn.Linear(time_emb_dim, dim_out * 2)) if time_emb_dim else None
        self.block1 = Block(dim,     dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb    = self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale  = dim_head ** -0.5; self.heads = heads
        hidden_dim  = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
    def forward(self, x):
        b, c, h, w = x.shape
        qkv   = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.heads, -1, h * w), qkv)
        sim   = torch.einsum('b h d i, b h d j -> b h i j', q * self.scale, k)
        attn  = sim.softmax(dim=-1)
        out   = torch.einsum('b h i j, b h d j -> b h d i', attn, v)
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
            return torch.mean(x, dim=(2, 3))        # (B, dim*4)
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
        self.steps  = steps; self.device = device
        betas       = torch.linspace(0.0001, 0.02, steps).to(device)
        alphas      = 1. - betas
        alphas_cp   = torch.cumprod(alphas, dim=0)
        self.betas  = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cp
        self.sa     = torch.sqrt(alphas_cp)
        self.so     = torch.sqrt(1. - alphas_cp)

    def q_sample(self, x0, t, noise=None):
        if noise is None: noise = torch.randn_like(x0)
        return (self.sa[t].view(-1,1,1,1)*x0 +
                self.so[t].view(-1,1,1,1)*noise), noise


# ── Training ──────────────────────────────────────────────────────────────────
def train_ae(model, loader, epochs):
    opt  = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    hist = []
    print(f"\n{'─'*55}\n  SimpleAutoencoder — {epochs} epochs\n{'─'*55}")
    for ep in range(1, epochs + 1):
        model.train(); acc = 0
        for imgs in tqdm(loader, desc=f"  AE {ep:4d}/{epochs}", leave=False):
            imgs = imgs.to(DEVICE); opt.zero_grad()
            recon, _ = model(imgs)
            loss = crit(recon, imgs)
            loss.backward(); opt.step()
            acc += loss.item()
        avg = acc / len(loader); hist.append(avg)
        if ep % 25 == 0 or ep == epochs:
            print(f"  epoch {ep:4d}   loss {avg:.5f}")
    torch.save({"state_dict": model.state_dict(), "epoch": epochs},
               os.path.join(CKPT_DIR, "ae.pth"))
    print(f"  Saved → {os.path.join(CKPT_DIR, 'ae.pth')}")
    return hist


def train_diffusion(model, diff, loader, epochs):
    opt  = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    hist = []
    print(f"\n{'─'*55}\n  UNet Diffusion — {epochs} epochs\n{'─'*55}")
    for ep in range(1, epochs + 1):
        model.train(); acc = 0
        for imgs in tqdm(loader, desc=f"  Diff {ep:4d}/{epochs}", leave=False):
            imgs = imgs.to(DEVICE)
            t    = torch.randint(0, diff.steps, (imgs.shape[0],), device=DEVICE).long()
            xn, noise = diff.q_sample(imgs, t)
            opt.zero_grad()
            loss = crit(model(xn, t), noise)
            loss.backward(); opt.step()
            acc += loss.item()
        avg = acc / len(loader); hist.append(avg)
        if ep % 25 == 0 or ep == epochs:
            print(f"  epoch {ep:4d}   loss {avg:.5f}")
    torch.save({"state_dict": model.state_dict(), "epoch": epochs},
               os.path.join(CKPT_DIR, "unet_diffusion.pth"))
    print(f"  Saved → {os.path.join(CKPT_DIR, 'unet_diffusion.pth')}")
    return hist


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ds      = SeismicDataset()
    loader  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                         drop_last=True, num_workers=2, pin_memory=True)
    diff    = DiffUtils()

    ae_params   = sum(p.numel() for p in SimpleAutoencoder().parameters())
    un_params   = sum(p.numel() for p in UNet().parameters())
    print(f"\n  SimpleAutoencoder params: {ae_params:,}")
    print(f"  UNet diffusion params   : {un_params:,}")

    ae   = SimpleAutoencoder().to(DEVICE)
    unet = UNet().to(DEVICE)

    ae_hist   = train_ae(ae, loader, AE_EPOCHS)
    diff_hist = train_diffusion(unet, diff, loader, DIFF_EPOCHS)

    # Loss plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.patch.set_facecolor('#111')
    for ax, hist, title, col in zip(
        axes,
        [ae_hist, diff_hist],
        [f"SimpleAutoencoder  [{AE_EPOCHS} ep]",
         f"UNet Diffusion  [{DIFF_EPOCHS} ep]"],
        ['#4fc3f7', '#a5d6a7'],
    ):
        ax.set_facecolor('#1a1a1a')
        ax.plot(hist, color=col, lw=2)
        ax.set_title(title, color='white', pad=6, fontsize=10)
        ax.set_xlabel("Epoch", color='#aaa')
        ax.set_ylabel("MSE Loss", color='#aaa')
        ax.tick_params(colors='#aaa')
        for sp in ax.spines.values(): sp.set_edgecolor('#444')
    fig.suptitle("Pretraining Loss — architectures from working_code/embeddings.ipynb",
                 color='white', fontsize=11, y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "pretraining_loss.png")
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor='#111')
    plt.close()
    print(f"\n  Loss plot → {path}")
    print(f"  Pretraining complete.  Checkpoints in {CKPT_DIR}/")


if __name__ == "__main__":
    main()
