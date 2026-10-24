"""
Evaluation script for DDPM-SR model.
Reconstructs exact architecture from sr_edsr.ipynb and evaluates
checkpoints_diffusion/diffusion_best.pth on the held-out val set
(odd-indexed inlines, per TRAIN_SPLIT_STRIDE=2).
"""
import os, math, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

VOLUME_PATH = '/datadisk/tupi/tupi/Sliepner/sliepner.npy'
CKPT_PATH   = '/datadisk/tupi/tupi/Sliepner/checkpoints_diffusion/diffusion_best.pth'
RESULTS_DIR = '/datadisk/tupi/tupi/Sliepner/eval_ddpm_sr_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

DIFFUSION_STEPS = 100
UNET_DIM        = 32
SCALE           = 2
BLUR_SIGMA      = 2.0

# ── Data loading ─────────────────────────────────────────────────────────────
print("Loading volume …")
vol = np.load(VOLUME_PATH).astype(np.float32)
norm_scale = np.percentile(np.abs(vol), 99.5)
vol = np.clip(vol / norm_scale, -1.0, 1.0)
print(f"  shape={vol.shape}  norm_scale={norm_scale:.4f}")

val_all = [i for i in range(vol.shape[0]) if i % 2 != 0]   # odd inlines
print(f"  Val inlines: {len(val_all)}")

# ── Exact degradation from sr_edsr.ipynb ─────────────────────────────────────
def get_blur_kernel(sigma, device):
    k_size = int(math.ceil(2 * sigma)) * 2 + 1
    k_half = (k_size - 1) * 0.5
    x = torch.linspace(-k_half, k_half, steps=k_size).to(device)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    k1d = pdf / pdf.sum()
    k2d = k1d.unsqueeze(1) @ k1d.unsqueeze(0)
    return k2d.unsqueeze(0).unsqueeze(0)

blur_k = get_blur_kernel(BLUR_SIGMA, DEVICE)

def apply_degradation(img, kernel, scale):
    pad = kernel.shape[-1] // 2
    blurred = F.conv2d(img, kernel, padding=pad)
    lr = F.interpolate(blurred, scale_factor=1/scale, mode='bicubic', align_corners=False)
    return lr

def degrade(hr_tensor):
    lr_small = apply_degradation(hr_tensor, blur_k, SCALE)
    lr_up = F.interpolate(lr_small, size=hr_tensor.shape[2:], mode='bicubic', align_corners=False)
    return lr_up, lr_small

# ── Architecture — exact match to sr_edsr.ipynb ───────────────────────────────
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
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
            s, sh = scale_shift
            x = x * (s + 1) + sh
        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if time_emb_dim else None
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
        self.scale    = dim_head ** -0.5
        self.heads    = heads
        hidden_dim    = dim_head * heads
        self.to_qkv   = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out   = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.heads, -1, h * w), qkv)
        q = q * self.scale
        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out  = torch.einsum('b h i j, b h d j -> b h d i', attn, v)
        out  = out.reshape(b, -1, h, w)
        return self.to_out(out) + x


class UNet(nn.Module):
    def __init__(self, dim=32, out_dim=1, channels=2):
        super().__init__()
        T = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, T), nn.GELU(), nn.Linear(T, T)
        )
        self.init_conv = nn.Conv2d(channels, dim, 7, padding=3)
        self.downs = nn.ModuleList([
            TimestepEmbedSequential(
                ResnetBlock(dim,    dim,    time_emb_dim=T),
                ResnetBlock(dim,    dim,    time_emb_dim=T)),
            TimestepEmbedSequential(
                ResnetBlock(dim,    dim*2,  time_emb_dim=T),
                ResnetBlock(dim*2,  dim*2,  time_emb_dim=T)),
            TimestepEmbedSequential(
                ResnetBlock(dim*2,  dim*4,  time_emb_dim=T),
                ResnetBlock(dim*4,  dim*4,  time_emb_dim=T)),
        ])
        self.down_samples = nn.ModuleList([
            nn.Conv2d(dim,   dim,   4, 2, 1),
            nn.Conv2d(dim*2, dim*2, 4, 2, 1),
            nn.Conv2d(dim*4, dim*4, 4, 2, 1),
        ])
        self.mid_block1 = ResnetBlock(dim*4, dim*4, time_emb_dim=T)
        self.mid_attn   = Attention(dim*4)
        self.mid_block2 = ResnetBlock(dim*4, dim*4, time_emb_dim=T)
        self.ups = nn.ModuleList([
            TimestepEmbedSequential(
                ResnetBlock(dim*8, dim*2,  time_emb_dim=T),
                ResnetBlock(dim*2, dim*2,  time_emb_dim=T)),
            TimestepEmbedSequential(
                ResnetBlock(dim*4, dim,    time_emb_dim=T),
                ResnetBlock(dim,   dim,    time_emb_dim=T)),
            TimestepEmbedSequential(
                ResnetBlock(dim*2, dim,    time_emb_dim=T),
                ResnetBlock(dim,   dim,    time_emb_dim=T)),
        ])
        self.up_samples = nn.ModuleList([
            nn.ConvTranspose2d(dim*4, dim*4, 4, 2, 1),
            nn.ConvTranspose2d(dim*2, dim*2, 4, 2, 1),
            nn.ConvTranspose2d(dim,   dim,   4, 2, 1),
        ])
        self.final_res_block = ResnetBlock(dim, dim, time_emb_dim=T)
        self.final_conv      = nn.Conv2d(dim, out_dim, 1)

    def forward(self, x, time, cond):
        x = torch.cat([x, cond], dim=1)
        t = self.time_mlp(time)
        x = self.init_conv(x)
        hs = []
        for block, ds in zip(self.downs, self.down_samples):
            x = block(x, t)
            hs.append(x)
            x = ds(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        for block, us in zip(self.ups, self.up_samples):
            x = us(x)
            h = hs.pop()
            if x.shape[2:] != h.shape[2:]:
                x = F.interpolate(x, size=h.shape[2:], mode='nearest')
            x = torch.cat((x, h), dim=1)
            x = block(x, t)
        return self.final_conv(self.final_res_block(x, t))


# ── Cosine DDPM schedule ─────────────────────────────────────────────────────
def make_cosine_schedule(steps=100, s=0.008, device='cpu'):
    xs   = torch.linspace(0, steps, steps + 1)
    acp  = torch.cos(((xs / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    acp  = (acp / acp[0]).to(device)
    betas      = (1 - acp[1:] / acp[:-1]).clamp(1e-4, 0.9999)
    alphas     = 1. - betas
    acp_prod   = torch.cumprod(alphas, 0)
    acp_prev   = F.pad(acp_prod[:-1], (1, 0), value=1.0)
    post_coef1 = betas * torch.sqrt(acp_prev) / (1. - acp_prod)
    post_coef2 = (1. - acp_prev) * torch.sqrt(alphas) / (1. - acp_prod)
    post_var   = betas * (1. - acp_prev) / (1. - acp_prod)
    return post_coef1, post_coef2, post_var


post_coef1, post_coef2, post_var = make_cosine_schedule(DIFFUSION_STEPS, device=DEVICE)


@torch.no_grad()
def ddpm_sample(model, cond):
    b, c, h, w = cond.shape
    img = torch.randn((b, 1, h, w), device=DEVICE)
    for i in tqdm(reversed(range(DIFFUSION_STEPS)), total=DIFFUSION_STEPS,
                  desc='  sample', leave=False):
        t = torch.full((b,), i, device=DEVICE, dtype=torch.long)
        pred = model(img, t, cond).clamp(-1., 1.)
        mean = post_coef1[i] * pred + post_coef2[i] * img
        if i == 0:
            img = mean
        else:
            img = mean + torch.sqrt(post_var[i]) * torch.randn_like(img)
    return img


# ── Load model ────────────────────────────────────────────────────────────────
print("Loading checkpoint …")
unet = UNet(dim=UNET_DIM, channels=2).to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
missing, unexpected = unet.load_state_dict(ckpt['model'], strict=True)
params = sum(p.numel() for p in unet.parameters())
print(f"  epoch={ckpt['epoch']}  best_psnr_stored={ckpt.get('best_psnr', 0):.2f} dB  "
      f"params={params/1e6:.2f}M  missing={len(missing)}  unexpected={len(unexpected)}")
unet.eval()

# ── Evaluation ────────────────────────────────────────────────────────────────
# Representative sample: every 8th odd inline (~16 inlines, balanced coverage)
eval_idx = val_all[::8]
print(f"\nEvaluating on {len(eval_idx)} representative val inlines (every 8th odd): {eval_idx}")
print(f"{'idx':>5}  {'bic_psnr':>10}  {'sr_psnr':>10}  {'gain':>8}  {'bic_ssim':>9}  {'sr_ssim':>8}")
print("─" * 60)

rows = []
for idx in eval_idx:
    hr = torch.tensor(vol[idx], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    lr_up, _ = degrade(hr)

    sr = ddpm_sample(unet, lr_up)
    if sr.shape[2:] != hr.shape[2:]:
        sr = F.interpolate(sr, size=hr.shape[2:], mode='bicubic', align_corners=False)
    sr = sr.clamp(-1., 1.)

    hr_np  = hr.squeeze().cpu().numpy()
    sr_np  = sr.squeeze().cpu().numpy()
    bic_np = lr_up.squeeze().cpu().numpy()

    dr = hr_np.max() - hr_np.min()   # per-sample range (matches training compute_metrics)
    p_sr   = psnr_metric(hr_np, sr_np,  data_range=dr)
    p_bic  = psnr_metric(hr_np, bic_np, data_range=dr)
    s_sr   = ssim_metric(hr_np, sr_np,  data_range=dr)
    s_bic  = ssim_metric(hr_np, bic_np, data_range=dr)

    # Also compute with fixed data_range=2.0 for cross-inline comparability
    p_sr_2   = psnr_metric(hr_np, sr_np,  data_range=2.0)
    p_bic_2  = psnr_metric(hr_np, bic_np, data_range=2.0)

    rows.append({
        'idx': idx, 'dr': dr,
        'bic_psnr': p_bic, 'sr_psnr': p_sr, 'gain': p_sr - p_bic,
        'bic_ssim': s_bic, 'sr_ssim': s_sr,
        'bic_psnr_2': p_bic_2, 'sr_psnr_2': p_sr_2, 'gain_2': p_sr_2 - p_bic_2,
    })
    print(f"{idx:>5}  {p_bic:>10.2f}  {p_sr:>10.2f}  {p_sr-p_bic:>+8.2f}  {s_bic:>9.4f}  {s_sr:>8.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
bic  = np.array([r['bic_psnr']   for r in rows])
sr   = np.array([r['sr_psnr']    for r in rows])
gain = np.array([r['gain']        for r in rows])
bic2 = np.array([r['bic_psnr_2'] for r in rows])
sr2  = np.array([r['sr_psnr_2']  for r in rows])
g2   = np.array([r['gain_2']     for r in rows])

print("─" * 60)
print(f"{'Mean':>5}  {bic.mean():>10.2f}  {sr.mean():>10.2f}  {gain.mean():>+8.2f}  "
      f"{np.mean([r['bic_ssim'] for r in rows]):>9.4f}  {np.mean([r['sr_ssim'] for r in rows]):>8.4f}")
print(f"{'Std':>5}  {bic.std():>10.2f}  {sr.std():>10.2f}  {gain.std():>8.2f}")

print("\n─── Per-sample data_range PSNR (matches training metric) ───")
print(f"  Bicubic  : {bic.mean():.2f} ± {bic.std():.2f} dB")
print(f"  DDPM-SR  : {sr.mean():.2f} ± {sr.std():.2f} dB")
print(f"  Gain     : {gain.mean():+.2f} ± {gain.std():.2f} dB")

print("\n─── Fixed data_range=2.0 PSNR (cross-inline comparable) ────")
print(f"  Bicubic  : {bic2.mean():.2f} ± {bic2.std():.2f} dB")
print(f"  DDPM-SR  : {sr2.mean():.2f} ± {sr2.std():.2f} dB")
print(f"  Gain     : {g2.mean():+.2f} ± {g2.std():.2f} dB")

# ── Save results ──────────────────────────────────────────────────────────────
summary = {
    'checkpoint': CKPT_PATH,
    'epoch': int(ckpt['epoch']),
    'n_eval_inlines': len(rows),
    'eval_inlines': eval_idx,
    'per_sample_dr': {
        'bicubic_mean': float(bic.mean()), 'bicubic_std': float(bic.std()),
        'ddpm_sr_mean': float(sr.mean()),  'ddpm_sr_std': float(sr.std()),
        'gain_mean':    float(gain.mean()), 'gain_std': float(gain.std()),
    },
    'fixed_dr2': {
        'bicubic_mean': float(bic2.mean()), 'bicubic_std': float(bic2.std()),
        'ddpm_sr_mean': float(sr2.mean()),  'ddpm_sr_std': float(sr2.std()),
        'gain_mean':    float(g2.mean()),   'gain_std': float(g2.std()),
    },
    'rows': rows,
}
out_path = os.path.join(RESULTS_DIR, 'results.json')
with open(out_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to {out_path}")
