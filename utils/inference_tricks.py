"""
Inference tricks comparison on v8 DDIM best checkpoint:
  - T=50  (baseline)
  - T=100
  - T=200
  - Ensemble N=5, T=50 (average of 5 runs with different starting noise)
Val inlines 200-248, plume box + full slice metrics.
"""
import sys, types, math, time, numpy as np, torch
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity   as _ssim

DEVICE    = 'cuda'
PLUME_XL  = (111, 401)
PLUME_T   = (431, 527)
RES_STD   = 0.041681
VAL_START = 200
CKPT      = '/datadisk/tupi/tupi/Sliepner/checkpoints_diffusion_interp_v8/diffusion_best.pth'

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

# ── Load v8 module ────────────────────────────────────────────────────────────
print("Loading v8 module …")
src8 = open('/datadisk/tupi/tupi/Sliepner/ddpm_interp_v8.py').read()
src8 = src8.replace('if __name__ == "__main__":', 'if False:')
mod8 = types.ModuleType('v8')
mod8.__file__ = '/datadisk/tupi/tupi/Sliepner/ddpm_interp_v8.py'
exec(compile(src8, 'ddpm_interp_v8.py', 'exec'), mod8.__dict__)

# ── Load weights once ─────────────────────────────────────────────────────────
print("Loading checkpoint …")
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
ep   = ckpt['epoch']
print(f"  ep{ep}  stored PSNR={ckpt['best_psnr']:.4f}")

def make_model(steps):
    unet = mod8.UNet(dim=mod8.C['UNET_DIM'], channels=mod8.C['INPUT_CHANNELS']).to(DEVICE)
    unet.load_state_dict(ckpt['model']); unet.eval()
    diff = mod8.Diffusion(unet, steps=steps, device=DEVICE).to(DEVICE)
    return diff

# ── Load volume ───────────────────────────────────────────────────────────────
print("Loading volume …")
vol   = np.load('/datadisk/tupi/tupi/Sliepner/sliepner.npy').astype(np.float32)
scale = np.percentile(np.abs(vol), 99.5)
vol   = np.clip(vol / scale, -1., 1.)
val_idx = np.arange(VAL_START, vol.shape[0])
print(f"  n_val={len(val_idx)}")

# ── Helper: single DDIM inference ────────────────────────────────────────────
def infer_once(sl, diff, seed=42):
    torch.manual_seed(seed)
    return mod8.infer_slice(sl, diff, RES_STD, DEVICE)

# ── Helper: ensemble inference ────────────────────────────────────────────────
def infer_ensemble(sl, diff, n=5):
    preds = [infer_once(sl, diff, seed=s) for s in range(n)]
    return np.mean(preds, axis=0)

# ── Experiment configs ────────────────────────────────────────────────────────
# (label, steps, ensemble_n)
experiments = [
    ('T=50  (baseline)',  50,  1),
    ('T=100',           100,  1),
    ('T=200',           200,  1),
    ('Ensemble N=5 T=50', 50, 5),
]

# Pre-build models (one per unique step count)
step_counts = sorted(set(e[1] for e in experiments))
models = {s: make_model(s) for s in step_counts}
print(f"Models built for T={step_counts}\n")

# ── Run all experiments ───────────────────────────────────────────────────────
all_results = {}   # label -> list of {full, plume}

for label, steps, n_ens in experiments:
    print(f"Running: {label} …")
    diff   = models[steps]
    rows   = []
    t0     = time.time()
    for k, vi in enumerate(val_idx):
        sl = vol[vi]
        if n_ens == 1:
            pred = infer_once(sl, diff, seed=42)
        else:
            pred = infer_ensemble(sl, diff, n=n_ens)
        mf = metrics_region(pred, sl)
        mp = metrics_region(pred, sl, xl=PLUME_XL, t=PLUME_T)
        rows.append({'full': mf, 'plume': mp})
        elapsed = time.time()-t0
        eta = elapsed/(k+1)*(len(val_idx)-k-1)
        print(f"  [{label}] inline {vi} ({k+1}/{len(val_idx)})  "
              f"plume_psnr={mp['psnr']:.2f}  ETA={eta:.0f}s", flush=True)
    all_results[label] = rows
    print(f"  Done in {time.time()-t0:.0f}s\n")

# ── Report ────────────────────────────────────────────────────────────────────
def avg(rows, zone, metric):
    return float(np.mean([r[zone][metric] for r in rows]))

METRICS = [('PSNR','psnr','.2f'), ('SSIM','ssim','.4f'),
           ('SNR', 'snr', '.2f'), ('NCC', 'ncc', '.4f'), ('MAE','mae','.5f')]

print("\n" + "="*80)
print(f"INFERENCE TRICKS — v8 DDIM ep{ep}")
print("="*80)

for zone, zlabel in [
    ('plume', 'PLUME BOX — crosslines 111–401, time 431–527'),
    ('full',  'FULL SLICE — 468 crosslines × 1001 time samples'),
]:
    print(f"\n  {zlabel}")
    labels = [e[0] for e in experiments]
    header = f"  {'Metric':6}  " + "  ".join(f"{l:>22}" for l in labels)
    print(header)
    print("  " + "-"*(len(header)-2))
    for mlabel, key, fmt in METRICS:
        vals = [avg(all_results[l], zone, key) for l in labels]
        best = max(vals) if key != 'mae' else min(vals)
        row  = f"  {mlabel:6}  "
        for v in vals:
            mark = " *" if abs(v-best)<1e-8 else "  "
            row += f"{v:{fmt}}{mark}             "
        print(row)

# Plume PSNR trajectory per inline — all 4 methods
print("\n  Per-inline PSNR in PLUME BOX:")
header = f"  {'inline':>6}  " + "  ".join(f"{e[0]:>22}" for e in experiments)
print(header)
for k, vi in enumerate(val_idx):
    vals = [all_results[l][k]['plume']['psnr'] for l,_,_ in experiments]
    best = max(vals)
    row  = f"  {vi:6}  "
    for v in vals:
        mark = " *" if abs(v-best)<1e-8 else "  "
        row += f"{v:8.2f}{mark}              "
    print(row)

print("\n* = best for that inline")
