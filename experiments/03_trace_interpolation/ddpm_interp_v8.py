#!/usr/bin/env python3
"""
ddpm_interp_v8.py — DDPM + DDIM deterministic inference for crossline trace interpolation.

Identical to v7 in architecture and training objective (x0-prediction DDPM noise loss
+ FFT + GDL), but validation and inference use DETERMINISTIC denoising (no noise added
at each reverse step — i.e. DDIM with η=0).

Motivation: 3-mode diagnostic on v7 best checkpoint (ep5) showed:
  - Stochastic (T=50, +noise)  → 36.14 dB  (-0.90 vs bilinear)  ← what v7 reports
  - Deterministic (T=50, no ε) → 37.31 dB  (+0.27 vs bilinear)  ← correct metric
  - Single-step  (x=0, t=0)    → 37.07 dB  (+0.03 vs bilinear)
Stochastic noise injection imposes a -1.17 dB penalty; the DDPM model has learned
superior representations that are only visible under deterministic inference.

v8 trains with deterministic val PSNR from epoch 1, so checkpoint selection and
learning-curve reporting are consistent with the correct inference mode.

Run:
  nohup python ddpm_interp_v8.py > ddpm_interp_v8.log 2>&1 &
"""
import os, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

torch.manual_seed(42); np.random.seed(42)

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG = {
    "VOLUME_PATH":        "/datadisk/tupi/tupi/Sliepner/sliepner.npy",
    "DEVICE":             "cuda" if torch.cuda.is_available() else "cpu",
    "BATCH_SIZE":         8,
    "EPOCHS":             100,
    "LR":                 1e-3,
    "LR_MIN":             1e-5,
    "GRAD_CLIP":          1.0,
    "DIFFUSION_STEPS":    50,
    "PATCH_H":            128,     # crossline patch height (even; stride-16 compatible)
    "PATCH_W":            256,     # time patch width (stride-16 compatible)
    "PATCHES_PER_SLICE":  10,      # random patches per training inline per epoch
    "UNET_DIM":           64,
    "INPUT_CHANNELS":     4,       # x_t | obs | mask | trend
    "SAVE_DIR":           "/datadisk/tupi/tupi/Sliepner/checkpoints_diffusion_interp_v8",
    "FFT_LOSS_WEIGHT":    0.1,
    "GDL_LOSS_WEIGHT":    1.0,
    "CLIP_PERCENTILE":    99.5,
    "VAL_FREQ":           5,       # validate every N epochs + final epoch
    "TRAIN_END":          200,     # inlines 0..TRAIN_END-1 → train; rest → val
    "RESUME":             False,
    "SEED":               42,
    # Pad dims for val full-slice inference (next mult of 16 above 468, 1001)
    "PAD_H":              12,      # 468 → 480
    "PAD_W":              7,       # 1001 → 1008
}
os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)
DEVICE = CONFIG["DEVICE"]
C = CONFIG
print(f"[ddpm_interp_v8-DDIM] device={DEVICE}  T={C['DIFFUSION_STEPS']}  "
      f"dim={C['UNET_DIM']}  patch={C['PATCH_H']}x{C['PATCH_W']}  "
      f"pps={C['PATCHES_PER_SLICE']}  BS={C['BATCH_SIZE']}  epochs={C['EPOCHS']}  "
      f"inference=deterministic-DDIM-eta0")


# ── Baseline helpers ──────────────────────────────────────────────────────────
def build_baseline_image(sl):
    """
    sl: (M_crosslines, N_time) numpy array, even crosslines are observed.
    Returns baseline: observed at even rows, linear interp at odd rows.
    """
    bl = sl.copy()
    M  = sl.shape[0]
    for r in range(1, M, 2):
        if r + 1 < M:
            bl[r] = 0.5 * (sl[r-1] + sl[r+1])
        else:
            bl[r] = sl[r-1]  # boundary extrapolation
    return bl


def build_channels(sl):
    """
    Build obs, mask, trend arrays from a full or patch inline slice.
    sl: (M, N) numpy, even rows = observed, odd rows = missing.
    Returns: obs (M,N), mask (M,N), trend (M,N)
    """
    obs   = build_baseline_image(sl)     # observed at even, linear interp at odd
    mask  = np.zeros_like(sl)
    mask[0::2] = 1.0                     # 1 at observed crosslines
    trend = np.zeros_like(sl)
    M = sl.shape[0]
    for r in range(1, M, 2):
        if r + 1 < M:
            trend[r] = sl[r+1] - sl[r-1]
    return obs, mask, trend


# ── Dataset ───────────────────────────────────────────────────────────────────
class TraceInterpDataset(Dataset):
    """
    Random (PATCH_H, PATCH_W) crops from training inline slices.
    Odd crosslines are treated as missing; even crosslines as observed.
    y0 (top-left row of patch) is always even to preserve the even=observed pattern.
    """
    def __init__(self, vol_norm, inline_indices, patch_h, patch_w,
                 patches_per_slice, res_std):
        self.vol  = vol_norm
        self.idx  = inline_indices
        self.pH   = patch_h
        self.pW   = patch_w
        self.pps  = patches_per_slice
        self.rst  = res_std
        M, N      = vol_norm.shape[1], vol_norm.shape[2]
        # largest even y0 such that patch fits: y0 + patch_h <= M
        self.max_y0 = ((M - patch_h) // 2) * 2
        self.max_x0 = N - patch_w

    def __len__(self):
        return len(self.idx) * self.pps

    def __getitem__(self, idx):
        inline_i = self.idx[idx % len(self.idx)]
        sl = self.vol[inline_i]                         # (468, 1001)

        # Random even y0, random x0
        y0 = int(np.random.randint(0, self.max_y0 // 2 + 1)) * 2
        x0 = int(np.random.randint(0, self.max_x0 + 1))
        patch = sl[y0 : y0 + self.pH, x0 : x0 + self.pW]   # (pH, pW)

        obs, mask, trend = build_channels(patch)
        residual = patch - obs                          # nonzero at odd rows only
        res_norm = residual / self.rst

        to_t = lambda a: torch.tensor(a, dtype=torch.float32).unsqueeze(0)
        target = to_t(res_norm)                                   # [1, pH, pW]
        cond   = torch.tensor(
            np.stack([obs, mask, trend], axis=0), dtype=torch.float32)  # [3, pH, pW]
        return target, cond


# ── Architecture (same UNet as v6) ───────────────────────────────────────────
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, x):
        half = self.dim // 2
        emb  = math.log(10000) / (half - 1)
        emb  = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb  = x[:,None] * emb[None,:]
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
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp      = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out*2)) if time_emb_dim else None
        self.block1   = Block(dim, dim_out, groups)
        self.block2   = Block(dim_out, dim_out, groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
    def forward(self, x, time_emb=None):
        ss = None
        if self.mlp is not None and time_emb is not None:
            te = self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
            ss = te.chunk(2, dim=1)
        h = self.block2(self.block1(x, ss))
        return h + self.res_conv(x)


class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x, emb):
        for layer in self:
            x = layer(x, emb) if isinstance(layer, ResnetBlock) else layer(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale  = dim_head**-0.5; self.heads = heads
        hd          = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hd*3, 1, bias=False)
        self.to_out = nn.Conv2d(hd, dim, 1)
    def forward(self, x):
        b, c, h, w = x.shape
        qkv  = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.heads, -1, h*w), qkv)
        q    = q * self.scale
        sim  = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out  = torch.einsum('b h i j, b h d j -> b h d i', attn, v)
        return self.to_out(out.reshape(b, -1, h, w)) + x


class UNet(nn.Module):
    """
    INPUT_CHANNELS=4: [x_t | obs | mask | trend].
    forward(x, time, cond) where cond=[obs|mask|trend] (3 channels).
    """
    def __init__(self, dim=64, out_dim=1, channels=4):
        super().__init__()
        T = dim * 4
        self.time_mlp   = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim,T), nn.GELU(), nn.Linear(T,T))
        self.init_conv  = nn.Conv2d(channels, dim, 7, padding=3)
        self.downs = nn.ModuleList([
            TimestepEmbedSequential(ResnetBlock(dim,   dim,   time_emb_dim=T), ResnetBlock(dim,   dim,   time_emb_dim=T)),
            TimestepEmbedSequential(ResnetBlock(dim,   dim*2, time_emb_dim=T), ResnetBlock(dim*2, dim*2, time_emb_dim=T)),
            TimestepEmbedSequential(ResnetBlock(dim*2, dim*4, time_emb_dim=T), ResnetBlock(dim*4, dim*4, time_emb_dim=T)),
        ])
        self.down_samples = nn.ModuleList([
            nn.Conv2d(dim,   dim,   4,2,1),
            nn.Conv2d(dim*2, dim*2, 4,2,1),
            nn.Conv2d(dim*4, dim*4, 4,2,1),
        ])
        self.mid_block1 = ResnetBlock(dim*4, dim*4, time_emb_dim=T)
        self.mid_attn   = Attention(dim*4)
        self.mid_block2 = ResnetBlock(dim*4, dim*4, time_emb_dim=T)
        self.ups = nn.ModuleList([
            TimestepEmbedSequential(ResnetBlock(dim*8, dim*2, time_emb_dim=T), ResnetBlock(dim*2, dim*2, time_emb_dim=T)),
            TimestepEmbedSequential(ResnetBlock(dim*4, dim,   time_emb_dim=T), ResnetBlock(dim,   dim,   time_emb_dim=T)),
            TimestepEmbedSequential(ResnetBlock(dim*2, dim,   time_emb_dim=T), ResnetBlock(dim,   dim,   time_emb_dim=T)),
        ])
        self.up_samples = nn.ModuleList([
            nn.ConvTranspose2d(dim*4, dim*4, 4,2,1),
            nn.ConvTranspose2d(dim*2, dim*2, 4,2,1),
            nn.ConvTranspose2d(dim,   dim,   4,2,1),
        ])
        self.final_res_block = ResnetBlock(dim, dim, time_emb_dim=T)
        self.final_conv      = nn.Conv2d(dim, out_dim, 1)

    def forward(self, x, time, cond):
        x = torch.cat([x, cond], dim=1)    # [B,4,H,W]
        t = self.time_mlp(time)
        x = self.init_conv(x)
        hs = []
        for blk, ds in zip(self.downs, self.down_samples):
            x = blk(x, t); hs.append(x); x = ds(x)
        x = self.mid_block1(x,t); x = self.mid_attn(x); x = self.mid_block2(x,t)
        for blk, us in zip(self.ups, self.up_samples):
            x = us(x); h = hs.pop()
            if x.shape[2:] != h.shape[2:]:
                x = F.interpolate(x, size=h.shape[2:], mode='nearest')
            x = blk(torch.cat((x, h), dim=1), t)
        return self.final_conv(self.final_res_block(x, t))


# ── DDPM ──────────────────────────────────────────────────────────────────────
class Diffusion(nn.Module):
    def __init__(self, model, steps=50, device='cpu'):
        super().__init__()
        self.model = model; self.steps = steps; self.device = device
        xs  = torch.linspace(0, steps, steps+1)
        s   = 0.008
        acp = torch.cos(((xs/steps)+s)/(1+s)*math.pi*0.5)**2
        acp = acp / acp[0]
        betas    = (1 - acp[1:]/acp[:-1]).clamp(1e-4, 0.9999)
        alphas   = 1. - betas
        acp      = torch.cumprod(alphas, 0)
        acp_prev = F.pad(acp[:-1], (1,0), value=1.0)
        self.register_buffer('sqrt_acp',    torch.sqrt(acp))
        self.register_buffer('sqrt_1m_acp', torch.sqrt(1.-acp))
        self.register_buffer('post_coef1',  betas * torch.sqrt(acp_prev) / (1.-acp))
        self.register_buffer('post_coef2',  (1.-acp_prev) * torch.sqrt(alphas) / (1.-acp))
        self.register_buffer('post_var',    betas * (1.-acp_prev) / (1.-acp))

    def q_sample(self, x0, t):
        noise = torch.randn_like(x0)
        xt    = self.sqrt_acp[t].view(-1,1,1,1)*x0 + self.sqrt_1m_acp[t].view(-1,1,1,1)*noise
        return xt, noise

    @torch.no_grad()
    def p_sample(self, x, t, t_idx, cond):
        pred = self.model(x, t, cond).clamp(-1., 1.)
        mean = self.post_coef1[t_idx]*pred + self.post_coef2[t_idx]*x
        return mean  # DDIM η=0: no noise added at any step

    @torch.no_grad()
    def sample(self, cond):
        b, _, h, w = cond.shape
        img = torch.randn((b, 1, h, w), device=self.device)
        for i in reversed(range(self.steps)):
            t   = torch.full((b,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t, i, cond)
        return img


# ── Loss helpers ──────────────────────────────────────────────────────────────
def compute_freq_loss(pred, target):
    fp = torch.fft.fftn(pred,   dim=(-2,-1))
    ft = torch.fft.fftn(target, dim=(-2,-1))
    return F.l1_loss(torch.log(torch.abs(fp)+1e-8), torch.log(torch.abs(ft)+1e-8))


def compute_gradient_loss(pred, target):
    pdx = torch.abs(pred[:,:,:,:-1]  - pred[:,:,:,1:])
    tdx = torch.abs(target[:,:,:,:-1] - target[:,:,:,1:])
    pdy = torch.abs(pred[:,:,:-1,:]  - pred[:,:,1:,:])
    tdy = torch.abs(target[:,:,:-1,:] - target[:,:,1:,:])
    return F.l1_loss(pdx, tdx) + F.l1_loss(pdy, tdy)


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(pred_np, target_np, data_range=2.0):
    dr   = data_range
    psnr = psnr_metric(target_np, pred_np, data_range=dr)
    ssim = ssim_metric(target_np, pred_np, data_range=dr)
    sig  = np.mean(target_np**2)
    nse  = np.mean((pred_np - target_np)**2)
    snr  = 10 * np.log10(sig / (nse + 1e-10))
    tc   = target_np - target_np.mean(); pc = pred_np - pred_np.mean()
    ncc  = float(np.sum(tc*pc) / (np.sqrt(np.sum(tc**2)*np.sum(pc**2)) + 1e-10))
    return psnr, ssim, snr, ncc


# ── Val inference on full slice ───────────────────────────────────────────────
@torch.no_grad()
def infer_slice(sl_norm, diffusion, res_std, device):
    """
    sl_norm: (468, 1001) float32 normalised inline slice.
    Returns: pred_full (468, 1001) reconstructed slice.
    """
    M, N   = sl_norm.shape
    pad_h  = C["PAD_H"]   # 12 → 480
    pad_w  = C["PAD_W"]   # 7  → 1008

    # Pad on bottom/right (preserves even=observed pattern at top-left)
    sl_pad = np.pad(sl_norm, ((0, pad_h), (0, pad_w)), mode='edge')

    obs, mask, trend = build_channels(sl_pad)
    cond = torch.tensor(
        np.stack([obs, mask, trend], axis=0), dtype=torch.float32
    ).unsqueeze(0).to(device)          # [1, 3, 480, 1008]

    pred_res_norm = diffusion.sample(cond)              # [1, 1, 480, 1008]
    pred_res = pred_res_norm.squeeze().cpu().numpy() * res_std  # (480, 1008)

    pred_full = np.clip(obs + pred_res, -1.0, 1.0)     # (480, 1008)
    return pred_full[:M, :N]                            # (468, 1001)


# ── Training ──────────────────────────────────────────────────────────────────
def main():
    # ── Load volume ──
    print("Loading seismic volume …")
    vol = np.load(C["VOLUME_PATH"]).astype(np.float32)
    norm_scale = np.percentile(np.abs(vol), C["CLIP_PERCENTILE"])
    vol = np.clip(vol / norm_scale, -1.0, 1.0)
    print(f"Volume shape: {vol.shape}  norm_scale={norm_scale:.4f}")

    n_inlines   = vol.shape[0]
    train_idx   = np.arange(0, C["TRAIN_END"])
    val_idx     = np.arange(C["TRAIN_END"], n_inlines)
    print(f"Train inlines: {len(train_idx)}  Val inlines: {len(val_idx)}")

    # ── Compute RES_STD from all training inlines ──
    print("Computing RES_STD from training set …")
    res_accum = []
    for i in train_idx:
        sl = vol[i]
        bl = build_baseline_image(sl)
        res_accum.append((sl[1::2] - bl[1::2]).ravel())
    res_std = float(np.std(np.concatenate(res_accum)))
    print(f"RES_STD={res_std:.6f}  (residual at missing crosslines)")

    # ── Compute linear baseline PSNR on full val set ──
    print("Computing linear baseline on val set …")
    bl_psnrs, bl_ssims, bl_nccs = [], [], []
    for i in val_idx:
        sl = vol[i]
        bl = build_baseline_image(sl)
        p, s, _, ncc = compute_metrics(bl, sl)
        bl_psnrs.append(p); bl_ssims.append(s); bl_nccs.append(ncc)
    bl_psnr_mean = np.mean(bl_psnrs)
    print(f"Linear baseline: PSNR {bl_psnr_mean:.2f}±{np.std(bl_psnrs):.2f} dB  "
          f"SSIM {np.mean(bl_ssims):.4f}  NCC {np.mean(bl_nccs):.4f}")

    # ── Datasets and loader ──
    train_ds = TraceInterpDataset(
        vol, train_idx,
        C["PATCH_H"], C["PATCH_W"], C["PATCHES_PER_SLICE"], res_std)
    print(f"Train dataset: {len(train_ds)} patches per epoch")

    loader = DataLoader(train_ds, batch_size=C["BATCH_SIZE"], shuffle=True,
                        drop_last=True, num_workers=4, pin_memory=True)

    # ── Model ──
    unet      = UNet(dim=C["UNET_DIM"], channels=C["INPUT_CHANNELS"]).to(DEVICE)
    diffusion = Diffusion(unet, steps=C["DIFFUSION_STEPS"], device=DEVICE).to(DEVICE)
    opt       = torch.optim.Adam(unet.parameters(), lr=C["LR"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=C["EPOCHS"], eta_min=C["LR_MIN"])

    params = sum(p.numel() for p in unet.parameters())
    print(f"UNet params: {params/1e6:.2f}M  (dim={C['UNET_DIM']}  ch={C['INPUT_CHANNELS']})")
    print(f"Loader: {len(loader)} batches/epoch  BS={C['BATCH_SIZE']}")

    ckpt_final = os.path.join(C["SAVE_DIR"], "diffusion_final.pth")
    ckpt_best  = os.path.join(C["SAVE_DIR"], "diffusion_best.pth")
    start_epoch = 1; best_psnr = -1.0

    history = {
        'epoch': [], 'mse': [], 'fft': [], 'gdl': [],
        'val_psnr': [], 'val_ssim': [], 'val_snr': [], 'val_ncc': [],
        'bl_psnr':  [], 'bl_ssim':  [], 'bl_snr':  [], 'bl_ncc':  [],
    }

    if C["RESUME"] and os.path.exists(ckpt_final):
        ckpt = torch.load(ckpt_final, map_location=DEVICE, weights_only=False)
        unet.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr   = ckpt.get('best_psnr', -1.0)
        if 'history' in ckpt: history = ckpt['history']
        print(f"Resumed from epoch {start_epoch-1}  best_psnr={best_psnr:.4f}")

    print(f"\nTraining: epochs {start_epoch}→{C['EPOCHS']}  "
          f"VAL_FREQ={C['VAL_FREQ']}  RES_STD={res_std:.6f}")
    print(f"Baseline PSNR to beat: {bl_psnr_mean:.2f} dB")

    for epoch in range(start_epoch, C["EPOCHS"] + 1):
        unet.train()
        t0     = time.time()
        totals = {'mse': 0., 'fft': 0., 'gdl': 0.}

        for target, cond in loader:
            target = target.to(DEVICE)   # [B, 1, pH, pW]  normalised residual
            cond   = cond.to(DEVICE)     # [B, 3, pH, pW]

            t_idx = torch.randint(0, C["DIFFUSION_STEPS"],
                                  (target.shape[0],), device=DEVICE).long()
            noisy, _ = diffusion.q_sample(target, t_idx)

            pred   = unet(noisy, t_idx, cond)
            l_mse  = F.mse_loss(pred, target)
            l_fft  = compute_freq_loss(pred, target)
            l_gdl  = compute_gradient_loss(pred, target)
            loss   = l_mse + C["FFT_LOSS_WEIGHT"]*l_fft + C["GDL_LOSS_WEIGHT"]*l_gdl

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), C["GRAD_CLIP"])
            opt.step()
            totals['mse'] += l_mse.item()
            totals['fft'] += l_fft.item()
            totals['gdl'] += l_gdl.item()

        n   = len(loader)
        avg = {k: v/n for k,v in totals.items()}
        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]

        do_val = (epoch % C["VAL_FREQ"] == 0) or (epoch == C["EPOCHS"])
        if not do_val:
            elapsed = time.time() - t0
            print(f"Ep{epoch:03d} | mse={avg['mse']:.4f} fft={avg['fft']:.4f} "
                  f"gdl={avg['gdl']:.4f} lr={cur_lr:.2e} | "
                  f"val skipped ({C['VAL_FREQ']}-ep freq) | {elapsed:.0f}s")
            history['epoch'].append(epoch)
            history['mse'].append(avg['mse']); history['fft'].append(avg['fft'])
            history['gdl'].append(avg['gdl'])
            continue

        # ── Validation: full-slice inference on all val inlines ──
        unet.eval()
        val_psnrs, val_ssims, val_snrs, val_nccs = [], [], [], []
        base_psnrs, base_ssims, base_snrs, base_nccs = [], [], [], []

        torch.manual_seed(42)
        with torch.no_grad():
            for vi in val_idx:
                sl = vol[vi]
                bl = build_baseline_image(sl)

                pred = infer_slice(sl, diffusion, res_std, DEVICE)

                vp, vs, vsnr, vncc = compute_metrics(pred, sl)
                bp, bs, bsnr, bncc = compute_metrics(bl,   sl)

                val_psnrs.append(vp);  val_ssims.append(vs)
                val_snrs.append(vsnr); val_nccs.append(vncc)
                base_psnrs.append(bp); base_ssims.append(bs)
                base_snrs.append(bsnr); base_nccs.append(bncc)

        avg_psnr  = np.mean(val_psnrs);  avg_ssim  = np.mean(val_ssims)
        avg_snr   = np.mean(val_snrs);   avg_ncc   = np.mean(val_nccs)
        avg_bpsnr = np.mean(base_psnrs); avg_bssim = np.mean(base_ssims)
        avg_bsnr  = np.mean(base_snrs);  avg_bncc  = np.mean(base_nccs)
        elapsed   = time.time() - t0

        gap = avg_psnr - avg_bpsnr
        print(f"Ep{epoch:03d} | mse={avg['mse']:.4f} fft={avg['fft']:.4f} "
              f"gdl={avg['gdl']:.4f} lr={cur_lr:.2e} | "
              f"PSNR={avg_psnr:.2f} SSIM={avg_ssim:.4f} SNR={avg_snr:.2f} NCC={avg_ncc:.4f} | "
              f"linear_bl: PSNR={avg_bpsnr:.2f} SSIM={avg_bssim:.4f} | "
              f"gap={gap:+.2f}dB | {elapsed:.0f}s")

        # Save 4-panel validation figure (4 representative val inlines)
        fig_indices = val_idx[np.linspace(0, len(val_idx)-1, 4, dtype=int)]
        n_rows = len(fig_indices)
        fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4*n_rows))
        if n_rows == 1: axes = [axes]
        torch.manual_seed(42)
        with torch.no_grad():
            for row, vi in enumerate(fig_indices):
                sl   = vol[vi]
                bl   = build_baseline_image(sl)
                pred = infer_slice(sl, diffusion, res_std, DEVICE)
                # Decimated for visualisation (zeros at missing)
                dec  = sl.copy(); dec[1::2] = 0.0

                vp, _, _, _ = compute_metrics(pred, sl)
                bp, _, _, _ = compute_metrics(bl,   sl)

                for col, (arr, title) in enumerate([
                    (dec,  f"Decimated (inline {vi})"),
                    (bl,   f"Linear baseline {bp:.2f}dB"),
                    (sl,   "Ground truth"),
                    (pred, f"DDPM-v8-DDIM {vp:.2f}dB"),
                ]):
                    ax = axes[row][col]
                    ax.imshow(arr.T, cmap='seismic', aspect='auto', vmin=-1, vmax=1)
                    ax.set_title(title, fontsize=9); ax.axis('off')

        plt.suptitle(f"v8-DDIM Ep{epoch:03d} | PSNR={avg_psnr:.2f}dB  "
                     f"linear_bl={avg_bpsnr:.2f}dB  gap={gap:+.2f}dB  "
                     f"SSIM={avg_ssim:.4f}  NCC={avg_ncc:.4f}", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(C["SAVE_DIR"], f"ep{epoch:03d}_val.png"),
                    dpi=100, bbox_inches='tight')
        plt.close()

        # Accumulate history
        history['epoch'].append(epoch)
        history['mse'].append(avg['mse']);  history['fft'].append(avg['fft'])
        history['gdl'].append(avg['gdl'])
        history['val_psnr'].append(avg_psnr);  history['val_ssim'].append(avg_ssim)
        history['val_snr'].append(avg_snr);    history['val_ncc'].append(avg_ncc)
        history['bl_psnr'].append(avg_bpsnr);  history['bl_ssim'].append(avg_bssim)
        history['bl_snr'].append(avg_bsnr);    history['bl_ncc'].append(avg_bncc)

        ckpt_data = {
            'epoch': epoch, 'model': unet.state_dict(), 'opt': opt.state_dict(),
            'best_psnr': best_psnr, 'config': C, 'history': history,
            'res_std': res_std, 'bl_psnr_baseline': bl_psnr_mean,
        }
        torch.save(ckpt_data, ckpt_final)

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            ckpt_data['best_psnr'] = best_psnr
            torch.save(ckpt_data, ckpt_best)
            print(f"  *** New best: {best_psnr:.4f} dB — checkpoint saved ***")

    print(f"\nTRAINING COMPLETE  best_psnr={best_psnr:.4f} dB  "
          f"linear_baseline={bl_psnr_mean:.2f} dB")
    print(f"Gap to baseline: {best_psnr - bl_psnr_mean:+.2f} dB")
    print(f"Checkpoints: {C['SAVE_DIR']}")


if __name__ == "__main__":
    main()
