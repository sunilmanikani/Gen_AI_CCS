#!/usr/bin/env python3
"""
ddpm_sr_v2.py — DDPM-SR with improvements borrowed from EDSR:
  - dim=64  (2× feature width, 4× parameters vs v1)
  - L1 reconstruction loss instead of MSE (robust to amplitude outliers)
  - res_scale=0.1 in ResnetBlock residuals (training stability for deeper/wider network)
  - Cosine annealing LR schedule (1e-3 → 1e-5 over 100 epochs)
  - Full 100-epoch run from scratch
  - Separate checkpoint dir: checkpoints_diffusion_v2/

Run:  nohup python ddpm_sr_v2.py > ddpm_sr_v2.log 2>&1 &
"""
import os, math, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

torch.manual_seed(42); np.random.seed(42)

# ── CONFIG (improvements over v1) ─────────────────────────────────────────────
CONFIG = {
    "VOLUME_PATH":       "/datadisk/tupi/tupi/Sliepner/sliepner.npy",
    "DEVICE":            "cuda" if torch.cuda.is_available() else "cpu",
    "BATCH_SIZE":        4,
    "EPOCHS":            100,
    "LR":                1e-3,
    "LR_MIN":            1e-5,          # cosine annealing floor
    "SCALE":             2,
    "DIFFUSION_STEPS":   100,
    "UNET_DIM":          64,            # v2: 64 (was 32) — 4× more parameters
    "RES_SCALE":         0.1,           # v2: residual scaling (EDSR-style)
    "SAVE_DIR":          "/datadisk/tupi/tupi/Sliepner/checkpoints_diffusion_v2",
    "BLUR_SIGMA":        2.0,
    "FFT_LOSS_WEIGHT":   0.1,
    "GDL_LOSS_WEIGHT":   1.0,
    "LOSS_TYPE":         "l1",          # v2: L1 main loss (was MSE)
    "SCHEDULE":          "cosine",
    "TRAIN_SPLIT_STRIDE": 2,
    "CLIP_PERCENTILE":   99.5,
    "RESUME":            True,          # continue from diffusion_final.pth
    "SEED":              42,
    "VAL_INDICES":       [5, 31, 62, 83, 100, 123],
}
os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)
DEVICE = CONFIG["DEVICE"]
print(f"[ddpm_sr_v2] device={DEVICE}  epochs={CONFIG['EPOCHS']}  dim={CONFIG['UNET_DIM']}"
      f"  loss={CONFIG['LOSS_TYPE']}  res_scale={CONFIG['RES_SCALE']}")

# ── Degradation helpers (identical to v1) ─────────────────────────────────────
def get_blur_kernel(sigma, device):
    k = int(math.ceil(2 * sigma)) * 2 + 1
    half = (k - 1) * 0.5
    x = torch.linspace(-half, half, steps=k).to(device)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    k1d = pdf / pdf.sum()
    k2d = (k1d.unsqueeze(1) @ k1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    return k2d

def apply_degradation(img, kernel, scale):
    pad = kernel.shape[-1] // 2
    blurred = F.conv2d(img, kernel, padding=pad)
    lr = F.interpolate(blurred, scale_factor=1/scale, mode='bicubic', align_corners=False)
    return lr

def compute_freq_loss(pred, target):
    fp = torch.fft.fftn(pred, dim=(-2,-1))
    ft = torch.fft.fftn(target, dim=(-2,-1))
    return F.l1_loss(torch.log(torch.abs(fp)+1e-8), torch.log(torch.abs(ft)+1e-8))

def compute_gradient_loss(pred, target):
    pdx = torch.abs(pred[:,:,:,:-1] - pred[:,:,:,1:])
    tdx = torch.abs(target[:,:,:,:-1] - target[:,:,:,1:])
    pdy = torch.abs(pred[:,:,:-1,:] - pred[:,:,1:,:])
    tdy = torch.abs(target[:,:,:-1,:] - target[:,:,1:,:])
    return F.l1_loss(pdx, tdx) + F.l1_loss(pdy, tdy)

# ── Dataset (identical to v1) ─────────────────────────────────────────────────
class SeismicSRDataset(Dataset):
    def __init__(self, volume_path, scale, split, blur_sigma, stride, clip_pct, device):
        vol = np.load(volume_path).astype(np.float32)
        self.norm_scale = np.percentile(np.abs(vol), clip_pct)
        vol = np.clip(vol / self.norm_scale, -1.0, 1.0)
        if split == 'train':
            self.slices = [vol[i] for i in range(0, vol.shape[0], stride)]
        else:
            self.slices = [vol[i] for i in range(vol.shape[0]) if i % stride != 0]
        self.scale = scale
        self.blur_kernel = get_blur_kernel(blur_sigma, 'cpu')
        print(f"  Dataset {split}: {len(self.slices)} inlines  norm_scale={self.norm_scale:.4f}")

    def __len__(self): return len(self.slices)

    def __getitem__(self, idx):
        hr = torch.tensor(self.slices[idx], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        lr_small = apply_degradation(hr, self.blur_kernel, self.scale)
        lr_up = F.interpolate(lr_small, size=hr.shape[2:], mode='bicubic', align_corners=False)
        return lr_up.squeeze(0), hr.squeeze(0), lr_small.squeeze(0)

# ── Architecture (v2: dim=64, res_scale=0.1) ──────────────────────────────────
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x[:,None] * emb[None,:]
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
            s, sh = scale_shift; x = x * (s+1) + sh
        return self.act(x)

class ResnetBlock(nn.Module):
    """ResnetBlock with optional EDSR-style residual scaling."""
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out*2)) if time_emb_dim else None
        self.block1   = Block(dim, dim_out, groups)
        self.block2   = Block(dim_out, dim_out, groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
    def forward(self, x, time_emb=None):
        ss = None
        if self.mlp is not None and time_emb is not None:
            te = self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
            ss = te.chunk(2, dim=1)
        h = self.block2(self.block1(x, ss))
        return h * self.res_scale + self.res_conv(x)

class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x, emb):
        for layer in self:
            x = layer(x, emb) if isinstance(layer, ResnetBlock) else layer(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5; self.heads = heads
        hd = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hd*3, 1, bias=False)
        self.to_out  = nn.Conv2d(hd, dim, 1)
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.heads, -1, h*w), qkv)
        q = q * self.scale
        sim  = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out  = torch.einsum('b h i j, b h d j -> b h d i', attn, v)
        return self.to_out(out.reshape(b, -1, h, w)) + x

class UNetV2(nn.Module):
    """UNet with configurable dim and res_scale."""
    def __init__(self, dim=64, out_dim=1, channels=2, res_scale=0.1):
        super().__init__()
        T = dim * 4
        RS = res_scale
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim,T), nn.GELU(), nn.Linear(T,T))
        self.init_conv = nn.Conv2d(channels, dim, 7, padding=3)
        self.downs = nn.ModuleList([
            TimestepEmbedSequential(
                ResnetBlock(dim,   dim,   time_emb_dim=T, res_scale=RS),
                ResnetBlock(dim,   dim,   time_emb_dim=T, res_scale=RS)),
            TimestepEmbedSequential(
                ResnetBlock(dim,   dim*2, time_emb_dim=T, res_scale=RS),
                ResnetBlock(dim*2, dim*2, time_emb_dim=T, res_scale=RS)),
            TimestepEmbedSequential(
                ResnetBlock(dim*2, dim*4, time_emb_dim=T, res_scale=RS),
                ResnetBlock(dim*4, dim*4, time_emb_dim=T, res_scale=RS)),
        ])
        self.down_samples = nn.ModuleList([
            nn.Conv2d(dim,   dim,   4,2,1),
            nn.Conv2d(dim*2, dim*2, 4,2,1),
            nn.Conv2d(dim*4, dim*4, 4,2,1),
        ])
        self.mid_block1 = ResnetBlock(dim*4, dim*4, time_emb_dim=T, res_scale=RS)
        self.mid_attn   = Attention(dim*4)
        self.mid_block2 = ResnetBlock(dim*4, dim*4, time_emb_dim=T, res_scale=RS)
        self.ups = nn.ModuleList([
            TimestepEmbedSequential(
                ResnetBlock(dim*8, dim*2, time_emb_dim=T, res_scale=RS),
                ResnetBlock(dim*2, dim*2, time_emb_dim=T, res_scale=RS)),
            TimestepEmbedSequential(
                ResnetBlock(dim*4, dim,   time_emb_dim=T, res_scale=RS),
                ResnetBlock(dim,   dim,   time_emb_dim=T, res_scale=RS)),
            TimestepEmbedSequential(
                ResnetBlock(dim*2, dim,   time_emb_dim=T, res_scale=RS),
                ResnetBlock(dim,   dim,   time_emb_dim=T, res_scale=RS)),
        ])
        self.up_samples = nn.ModuleList([
            nn.ConvTranspose2d(dim*4, dim*4, 4,2,1),
            nn.ConvTranspose2d(dim*2, dim*2, 4,2,1),
            nn.ConvTranspose2d(dim,   dim,   4,2,1),
        ])
        self.final_res_block = ResnetBlock(dim, dim, time_emb_dim=T, res_scale=RS)
        self.final_conv      = nn.Conv2d(dim, out_dim, 1)

    def forward(self, x, time, cond):
        x = torch.cat([x, cond], dim=1)
        t = self.time_mlp(time)
        x = self.init_conv(x)
        hs = []
        for blk, ds in zip(self.downs, self.down_samples):
            x = blk(x, t); hs.append(x); x = ds(x)
        x = self.mid_block1(x,t); x = self.mid_attn(x); x = self.mid_block2(x,t)
        for blk, us in zip(self.ups, self.up_samples):
            x = us(x); h = hs.pop()
            if x.shape[2:] != h.shape[2:]: x = F.interpolate(x, size=h.shape[2:], mode='nearest')
            x = blk(torch.cat((x,h), dim=1), t)
        return self.final_conv(self.final_res_block(x, t))

# ── DDPM (identical to v1) ────────────────────────────────────────────────────
class Diffusion(nn.Module):
    def __init__(self, model, steps=100, device='cpu'):
        super().__init__()
        self.model = model; self.steps = steps; self.device = device
        xs  = torch.linspace(0, steps, steps+1)
        s   = 0.008
        acp = torch.cos(((xs/steps)+s)/(1+s)*math.pi*0.5)**2
        acp = (acp / acp[0])
        betas  = (1 - acp[1:]/acp[:-1]).clamp(1e-4, 0.9999)
        alphas = 1. - betas
        acp    = torch.cumprod(alphas, 0)
        acp_prev = F.pad(acp[:-1], (1,0), value=1.0)
        self.register_buffer('sqrt_acp',    torch.sqrt(acp))
        self.register_buffer('sqrt_1m_acp', torch.sqrt(1.-acp))
        self.register_buffer('post_coef1',  betas * torch.sqrt(acp_prev) / (1.-acp))
        self.register_buffer('post_coef2',  (1.-acp_prev) * torch.sqrt(alphas) / (1.-acp))
        self.register_buffer('post_var',    betas * (1.-acp_prev) / (1.-acp))

    def q_sample(self, x0, t):
        noise = torch.randn_like(x0)
        xt = self.sqrt_acp[t].view(-1,1,1,1)*x0 + self.sqrt_1m_acp[t].view(-1,1,1,1)*noise
        return xt, noise

    @torch.no_grad()
    def p_sample(self, x, t, t_idx, cond):
        pred = self.model(x, t, cond).clamp(-1.,1.)
        mean = self.post_coef1[t_idx]*pred + self.post_coef2[t_idx]*x
        if t_idx == 0: return mean
        return mean + torch.sqrt(self.post_var[t_idx]) * torch.randn_like(x)

    @torch.no_grad()
    def sample(self, cond):
        b,c,h,w = cond.shape
        img = torch.randn((b,1,h,w), device=self.device)
        for i in reversed(range(self.steps)):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t, i, cond)
        return img

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(sr, hr):
    s = sr.detach().cpu().squeeze().numpy()
    h = hr.detach().cpu().squeeze().numpy()
    dr = h.max() - h.min()
    return psnr_metric(h, s, data_range=dr), ssim_metric(h, s, data_range=dr)

# ── Training ──────────────────────────────────────────────────────────────────
def main():
    C = CONFIG
    print("Loading datasets …")
    train_ds = SeismicSRDataset(C["VOLUME_PATH"], C["SCALE"], 'train',
                                C["BLUR_SIGMA"], C["TRAIN_SPLIT_STRIDE"], C["CLIP_PERCENTILE"], DEVICE)
    val_ds   = SeismicSRDataset(C["VOLUME_PATH"], C["SCALE"], 'val',
                                C["BLUR_SIGMA"], C["TRAIN_SPLIT_STRIDE"], C["CLIP_PERCENTILE"], DEVICE)
    loader   = DataLoader(train_ds, batch_size=C["BATCH_SIZE"], shuffle=True, drop_last=True,
                          num_workers=4, pin_memory=True)
    val_indices = [min(v, len(val_ds)-1) for v in C["VAL_INDICES"]]

    unet      = UNetV2(dim=C["UNET_DIM"], channels=2, res_scale=C["RES_SCALE"]).to(DEVICE)
    diffusion = Diffusion(unet, steps=C["DIFFUSION_STEPS"], device=DEVICE).to(DEVICE)
    opt       = torch.optim.Adam(unet.parameters(), lr=C["LR"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C["EPOCHS"], eta_min=C["LR_MIN"])

    start_epoch = 1; best_psnr = -1.0
    history = {'epoch': [], 'lr': [], 'main': [], 'fft': [], 'gdl': [],
               'val_psnr': [], 'val_ssim': []}

    if C["RESUME"]:
        # Resume from latest epoch (final.pth), not best — optimizer state must be current
        ckpt_path = os.path.join(C["SAVE_DIR"], "diffusion_final.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(C["SAVE_DIR"], "diffusion_best.pth")  # fallback
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            unet.load_state_dict(ckpt['model'])
            opt.load_state_dict(ckpt['opt'])
            start_epoch = ckpt['epoch'] + 1
            best_psnr   = ckpt.get('best_psnr', -1.0)
            if 'history' in ckpt: history = ckpt['history']
            print(f"Resumed from epoch {start_epoch-1}  best_psnr={best_psnr:.4f}")

    params = sum(p.numel() for p in unet.parameters())
    print(f"UNetV2 params: {params/1e6:.2f}M  (v1 was 2.73M)")
    print(f"Improvements: dim={C['UNET_DIM']} | loss={C['LOSS_TYPE']} | "
          f"res_scale={C['RES_SCALE']} | cosine LR {C['LR']}→{C['LR_MIN']}")
    print(f"Training epochs {start_epoch} → {C['EPOCHS']}")

    # Advance scheduler to match resume epoch
    for _ in range(start_epoch - 1):
        scheduler.step()

    loss_fn = F.l1_loss if C["LOSS_TYPE"] == "l1" else F.mse_loss

    for epoch in range(start_epoch, C["EPOCHS"]+1):
        unet.train()
        t0 = time.time()
        totals = {'main': 0., 'fft': 0., 'gdl': 0.}
        current_lr = scheduler.get_last_lr()[0] if epoch > 1 else C["LR"]
        pbar = tqdm(loader, desc=f"Ep{epoch:03d} lr={current_lr:.2e}", leave=False, ncols=95)
        for lr_up, hr, _ in pbar:
            lr_up, hr = lr_up.to(DEVICE), hr.to(DEVICE)
            t = torch.randint(0, C["DIFFUSION_STEPS"], (hr.shape[0],), device=DEVICE).long()
            noisy_hr, _ = diffusion.q_sample(hr, t)
            if noisy_hr.shape != lr_up.shape:
                lr_up = F.interpolate(lr_up, size=noisy_hr.shape[2:], mode='bilinear', align_corners=False)
            pred   = unet(noisy_hr, t, lr_up)
            l_main = loss_fn(pred, hr)
            l_fft  = compute_freq_loss(pred, hr)
            l_gdl  = compute_gradient_loss(pred, hr)
            loss   = l_main + C["FFT_LOSS_WEIGHT"]*l_fft + C["GDL_LOSS_WEIGHT"]*l_gdl
            opt.zero_grad(); loss.backward(); opt.step()
            totals['main'] += l_main.item(); totals['fft'] += l_fft.item(); totals['gdl'] += l_gdl.item()
            pbar.set_postfix(**{C["LOSS_TYPE"]: f"{l_main.item():.4f}",
                                'fft': f"{l_fft.item():.4f}", 'gdl': f"{l_gdl.item():.4f}"})
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        n = len(loader)
        avg = {k: v/n for k,v in totals.items()}

        # Validate — fixed seed so noise is identical every epoch; PSNR reflects model quality not luck
        unet.eval()
        val_psnrs, val_ssims = [], []
        fig_rows = []
        torch.manual_seed(42)
        with torch.no_grad():
            for v_idx in val_indices:
                lr_up, hr, _ = val_ds[v_idx]
                lr_up = lr_up.unsqueeze(0).to(DEVICE); hr = hr.unsqueeze(0).to(DEVICE)
                sr = diffusion.sample(lr_up)
                if sr.shape != hr.shape:
                    sr = F.interpolate(sr, size=hr.shape[2:], mode='bilinear', align_corners=False)
                p, s = compute_metrics(sr, hr)
                val_psnrs.append(p); val_ssims.append(s)
                fig_rows.append((lr_up.squeeze().cpu(), hr.squeeze().cpu(), sr.squeeze().cpu(), p))

        avg_psnr = np.mean(val_psnrs); avg_ssim = np.mean(val_ssims)
        elapsed  = time.time() - t0

        print(f"Ep{epoch:03d} | {C['LOSS_TYPE']}={avg['main']:.4f} fft={avg['fft']:.4f} "
              f"gdl={avg['gdl']:.4f} lr={current_lr:.2e} | "
              f"val_psnr={avg_psnr:.2f} ssim={avg_ssim:.4f} | {elapsed:.0f}s")

        # Save validation figure
        n_rows = len(fig_rows)
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4*n_rows))
        if n_rows == 1: axes = [axes]
        for row, (lr_np, hr_np, sr_np, p) in enumerate(fig_rows):
            for col, (arr, title) in enumerate([
                (lr_np, "LR_up (bicubic)"),
                (hr_np, "Target HR"),
                (sr_np, f"DDPM-SR v2  PSNR={p:.2f}dB"),
            ]):
                ax = axes[row][col]
                ax.imshow(arr.T, cmap='seismic', aspect='auto', vmin=-1, vmax=1)
                ax.set_title(title, fontsize=9); ax.axis('off')
        plt.suptitle(f"v2 Ep{epoch:03d} dim={C['UNET_DIM']} {C['LOSS_TYPE']} res_scale={C['RES_SCALE']} | "
                     f"avg PSNR={avg_psnr:.2f}dB SSIM={avg_ssim:.4f}", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(C["SAVE_DIR"], f"ep{epoch:03d}_val.png"), dpi=100)
        plt.close()

        # History
        history['epoch'].append(epoch)
        history['lr'].append(float(current_lr))
        history['main'].append(avg['main']); history['fft'].append(avg['fft']); history['gdl'].append(avg['gdl'])
        history['val_psnr'].append(float(avg_psnr)); history['val_ssim'].append(float(avg_ssim))
        with open(os.path.join(C["SAVE_DIR"], "history.json"), 'w') as f:
            json.dump(history, f, indent=2)

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save({'epoch': epoch, 'model': unet.state_dict(), 'opt': opt.state_dict(),
                        'best_psnr': best_psnr, 'history': history, 'config': C},
                       os.path.join(C["SAVE_DIR"], "diffusion_best.pth"))
            print(f"  *** New best: {best_psnr:.4f} dB — checkpoint saved ***")

        torch.save({'epoch': epoch, 'model': unet.state_dict(), 'opt': opt.state_dict(),
                    'best_psnr': best_psnr, 'history': history, 'config': C},
                   os.path.join(C["SAVE_DIR"], "diffusion_final.pth"))

    print(f"\nTraining complete. Best val PSNR: {best_psnr:.4f} dB")
    print(f"Best checkpoint: {os.path.join(C['SAVE_DIR'], 'diffusion_best.pth')}")

if __name__ == "__main__":
    main()
