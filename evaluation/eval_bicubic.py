"""
Compute all metrics for bicubic baseline on val inlines 200-248.
Bicubic: cubic spline interpolation of missing odd crosslines from observed even crosslines.
"""
import math, numpy as np
from scipy.interpolate import CubicSpline
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity   as _ssim

PLUME_XL  = (111, 401)
PLUME_T   = (431, 527)
VAL_START = 200

def bicubic_interp(sl):
    M, N = sl.shape
    out  = sl.copy()
    even = np.arange(0, M, 2)        # observed crosslines
    for col in range(N):
        cs = CubicSpline(even, sl[even, col], bc_type='not-a-knot')
        for r in range(1, M, 2):
            out[r, col] = np.clip(cs(r), -1., 1.)
    return out

def bilinear_interp(sl):
    bl = sl.copy(); M = sl.shape[0]
    for r in range(1, M, 2):
        bl[r] = 0.5*(sl[r-1] + sl[r+1]) if r+1 < M else sl[r-1]
    return bl

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

print("Loading volume …")
vol   = np.load('/datadisk/tupi/tupi/Sliepner/sliepner.npy').astype(np.float32)
scale = np.percentile(np.abs(vol), 99.5)
vol   = np.clip(vol / scale, -1., 1.)
val_idx = np.arange(VAL_START, vol.shape[0])
print(f"  n_val={len(val_idx)}")

rows_bl, rows_bc = [], []

print("Computing baselines …")
for k, vi in enumerate(val_idx):
    sl = vol[vi]
    bl = bilinear_interp(sl)
    bc = bicubic_interp(sl)
    rows_bl.append({'full': metrics_region(bl, sl),
                    'plume': metrics_region(bl, sl, xl=PLUME_XL, t=PLUME_T)})
    rows_bc.append({'full': metrics_region(bc, sl),
                    'plume': metrics_region(bc, sl, xl=PLUME_XL, t=PLUME_T)})
    print(f"  inline {vi} ({k+1}/{len(val_idx)})  "
          f"bilinear_psnr={rows_bl[-1]['full']['psnr']:.2f}  "
          f"bicubic_psnr={rows_bc[-1]['full']['psnr']:.2f}", flush=True)

def avg(rows, zone, metric):
    return float(np.mean([r[zone][metric] for r in rows]))

METRICS = [('PSNR','psnr','.2f','dB'),
           ('SSIM','ssim','.4f',''),
           ('SNR', 'snr', '.2f','dB'),
           ('NCC', 'ncc', '.4f',''),
           ('MAE', 'mae', '.5f','')]

print("\n" + "="*60)
print("BICUBIC vs BILINEAR BASELINE")
print(f"Val inlines {VAL_START}-{val_idx[-1]} ({len(val_idx)} inlines)")
print("="*60)
for zone, zlabel in [('full','FULL SLICE'), ('plume','PLUME BOX')]:
    print(f"\n  {zlabel}")
    print(f"  {'Metric':8} {'Bilinear':>12} {'Bicubic':>12} {'Δ':>10}")
    print("  " + "-"*44)
    for mlabel, key, fmt, unit in METRICS:
        b = avg(rows_bl, zone, key)
        c = avg(rows_bc, zone, key)
        d = c - b
        sign = '+' if d >= 0 else ''
        print(f"  {mlabel+unit:8} {b:{fmt}}     {c:{fmt}}   {sign}{d:{fmt}}")
