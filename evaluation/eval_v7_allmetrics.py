"""
Full-metric evaluation of v7 DDPM final checkpoint with DDIM deterministic inference.
Reports PSNR, SSIM, SNR, NCC, MAE — full slice + plume box.
Uses v8's DDIM p_sample (no noise) applied to v7's trained weights.
"""
import sys, types, math, time, numpy as np, torch
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity   as _ssim

DEVICE    = 'cuda'
PLUME_XL  = (111, 401)
PLUME_T   = (431, 527)
RES_STD   = 0.041681
VAL_START = 200

def metrics_region(pred, gt, xl=None, t=None):
    if xl is not None:
        pred = pred[xl[0]:xl[1], t[0]:t[1]]
        gt   = gt  [xl[0]:xl[1], t[0]:t[1]]
    dr   = 2.0
    psnr = _psnr(gt, pred, data_range=dr)
    ssim = _ssim(gt, pred, data_range=dr)
    sig  = float(np.mean(gt**2))
    nse  = float(np.mean((pred-gt)**2))
    snr  = 10*math.log10(sig / (nse + 1e-12))
    tc = gt - gt.mean(); pc = pred - pred.mean()
    ncc = float(np.sum(tc*pc) / (math.sqrt(float(np.sum(tc**2))*float(np.sum(pc**2))) + 1e-12))
    mae = float(np.mean(np.abs(pred-gt)))
    return dict(psnr=psnr, ssim=ssim, snr=snr, ncc=ncc, mae=mae)

# Load v8 module for architecture + helpers (same UNet/Diffusion as v7)
print("Loading v8 module (architecture) …")
src8 = open('/datadisk/tupi/tupi/Sliepner/ddpm_interp_v8.py').read()
src8 = src8.replace('if __name__ == "__main__":', 'if False:')
mod8 = types.ModuleType('v8')
mod8.__file__ = '/datadisk/tupi/tupi/Sliepner/ddpm_interp_v8.py'
exec(compile(src8, 'ddpm_interp_v8.py', 'exec'), mod8.__dict__)

# Load v7 weights into v8's DDIM Diffusion (deterministic p_sample)
print("Loading v7 final checkpoint …")
ckpt_v7f = torch.load(
    '/datadisk/tupi/tupi/Sliepner/checkpoints_diffusion_interp_v7/diffusion_final.pth',
    map_location=DEVICE, weights_only=False)
unet_v7f = mod8.UNet(dim=64, channels=4).to(DEVICE)
unet_v7f.load_state_dict(ckpt_v7f['model']); unet_v7f.eval()
diff_v7f = mod8.Diffusion(unet_v7f, steps=50, device=DEVICE).to(DEVICE)
ep_v7f   = ckpt_v7f.get('epoch', '?')
print(f"  v7 final: ep{ep_v7f}")

print("Loading v7 best checkpoint …")
ckpt_v7b = torch.load(
    '/datadisk/tupi/tupi/Sliepner/checkpoints_diffusion_interp_v7/diffusion_best.pth',
    map_location=DEVICE, weights_only=False)
unet_v7b = mod8.UNet(dim=64, channels=4).to(DEVICE)
unet_v7b.load_state_dict(ckpt_v7b['model']); unet_v7b.eval()
diff_v7b = mod8.Diffusion(unet_v7b, steps=50, device=DEVICE).to(DEVICE)
ep_v7b   = ckpt_v7b.get('epoch', '?')
print(f"  v7 best:  ep{ep_v7b}")

# Load v8 best for reference
print("Loading v8 best checkpoint …")
ckpt_v8  = torch.load(
    '/datadisk/tupi/tupi/Sliepner/checkpoints_diffusion_interp_v8/diffusion_best.pth',
    map_location=DEVICE, weights_only=False)
unet_v8  = mod8.UNet(dim=64, channels=4).to(DEVICE)
unet_v8.load_state_dict(ckpt_v8['model']); unet_v8.eval()
diff_v8  = mod8.Diffusion(unet_v8, steps=50, device=DEVICE).to(DEVICE)
ep_v8    = ckpt_v8.get('epoch', '?')
print(f"  v8 best:  ep{ep_v8}  stored_psnr={ckpt_v8['best_psnr']:.4f}")

# Load v7unet
print("Loading v7unet best checkpoint …")
srcu = open('/datadisk/tupi/tupi/Sliepner/ddpm_interp_v7unet.py').read()
srcu = srcu.replace('if __name__ == "__main__":', 'if False:')
modu = types.ModuleType('v7unet')
modu.__file__ = '/datadisk/tupi/tupi/Sliepner/ddpm_interp_v7unet.py'
exec(compile(srcu, 'ddpm_interp_v7unet.py', 'exec'), modu.__dict__)

ckpt_un = torch.load(
    '/datadisk/tupi/tupi/Sliepner/checkpoints_diffusion_interp_v7unet/unet_best.pth',
    map_location=DEVICE, weights_only=False)
unet_un = modu.UNet(dim=modu.C['UNET_DIM'], channels=modu.C['INPUT_CHANNELS']).to(DEVICE)
unet_un.load_state_dict(ckpt_un['model']); unet_un.eval()
ep_un   = ckpt_un.get('epoch', '?')
print(f"  v7unet:   ep{ep_un}  stored_psnr={ckpt_un['best_psnr']:.4f}")

# Volume
print("Loading volume …")
vol   = np.load('/datadisk/tupi/tupi/Sliepner/sliepner.npy').astype(np.float32)
scale = np.percentile(np.abs(vol), 99.5)
vol   = np.clip(vol / scale, -1., 1.)
val_idx = np.arange(VAL_START, vol.shape[0])
print(f"  n_val={len(val_idx)}\n")

# Inference
experiments = [
    ('Bilinear',              None,      None),
    (f'v7unet  ep{ep_un}',   'unet',    None),
    (f'v7 DDIM ep{ep_v7b}',  'diff',    diff_v7b),
    (f'v7 DDIM ep{ep_v7f}',  'diff',    diff_v7f),
    (f'v8 DDIM ep{ep_v8}',   'diff',    diff_v8),
]

all_results = {e[0]: [] for e in experiments}

print(f"Running inference on {len(val_idx)} val inlines …")
torch.manual_seed(42)
t0 = time.time()
for k, vi in enumerate(val_idx):
    sl = vol[vi]
    bl = mod8.build_baseline_image(sl)
    pu = modu.infer_slice(sl, unet_un, RES_STD, DEVICE)

    for label, mode, model in experiments:
        if mode is None:
            pred = bl
        elif mode == 'unet':
            pred = pu
        else:
            pred = mod8.infer_slice(sl, model, RES_STD, DEVICE)
        mf = metrics_region(pred, sl)
        mp = metrics_region(pred, sl, xl=PLUME_XL, t=PLUME_T)
        all_results[label].append({'full': mf, 'plume': mp})

    elapsed = time.time()-t0
    eta = elapsed/(k+1)*(len(val_idx)-k-1)
    print(f"  inline {vi} ({k+1}/{len(val_idx)})  ETA={eta:.0f}s", flush=True)

# Report
def avg(rows, zone, metric):
    return float(np.mean([r[zone][metric] for r in rows]))

METRICS = [('PSNR','psnr','.2f','dB'),
           ('SSIM','ssim','.4f',''),
           ('SNR', 'snr', '.2f','dB'),
           ('NCC', 'ncc', '.4f',''),
           ('MAE', 'mae', '.5f','')]

labels = [e[0] for e in experiments]

print("\n" + "="*90)
print("ALL-METRIC COMPARISON — Bilinear | v7unet | v7 DDIM (best/final) | v8 DDIM (best)")
print(f"Val inlines {VAL_START}-{val_idx[-1]} ({len(val_idx)} inlines)")
print("="*90)

for zone, zlabel in [
    ('plume', 'PLUME BOX — crosslines 111–401, time 431–527'),
    ('full',  'FULL SLICE — 468 crosslines × 1001 time samples'),
]:
    print(f"\n  {zlabel}")
    col_w = 16
    hdr = f"  {'':8}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(hdr)
    print("  " + "-"*(8 + col_w*len(labels)))
    for mlabel, key, fmt, unit in METRICS:
        vals = [avg(all_results[l], zone, key) for l in labels]
        best = min(vals) if key == 'mae' else max(vals)
        row  = f"  {mlabel+unit:8}"
        for v in vals:
            marker = " ◄" if abs(v-best) < 1e-8 else "  "
            row   += f"{v:{fmt}}{marker}".rjust(col_w)
        print(row)

print(f"\nTotal time: {time.time()-t0:.0f}s")
