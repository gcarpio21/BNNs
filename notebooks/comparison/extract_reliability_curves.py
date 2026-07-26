"""
Extract calibration curves from the 2-panel reliability_povertymap.png,
replot as a single merged 6x4.5" figure matching poster style,
and save a diff overlay to assess extraction accuracy.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm as _norm

ROOT = Path(__file__).resolve().parent
while ROOT != ROOT.parent and not (ROOT / "shared").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

OUT = str(ROOT / "submission" / "CPS_Poster_Template" / "figures")
SRC = str(ROOT / "notebooks" / "comparison" / "results" / "figures")
IMG_PATH = f"{SRC}/povertymap_calibration_curves.png"

plt.rcParams.update({"font.family": "serif", "font.size": 11,
                     "savefig.dpi": 180, "savefig.bbox": "tight"})

CONF_LEVELS = np.linspace(0.05, 0.95, 20)

# ── 1. Load image ──────────────────────────────────────────────────────────────
img = Image.open(IMG_PATH).convert('RGBA')
arr = np.asarray(img)
H, W = arr.shape[:2]
print(f"Image: {W}x{H}")

# ── 2. Subplot bounds ──────────────────────────────────────────────────────────
# Left subplot (ID): cols 100-851, rows 115-649
# Data area (inside spines):
#   x: 101-850  (750px wide, maps to data [0,1])
#   y: 117-647  (531px tall, maps to data [0,1])
#   y=117 → data=1.0 (top of axes), y=647 → data=0.0 (bottom of axes)
left_data  = {'x0': 102, 'x1': 850, 'y_bottom': 647, 'y_top': 117}
right_data = {'x0': 988, 'x1': 1737, 'y_bottom': 647, 'y_top': 117}

def pixel_y_to_data(py, bounds):
    """Convert pixel y (0=top of data, H-1=bottom) to data y [0,1]."""
    return (bounds['y_bottom'] - py) / (bounds['y_bottom'] - bounds['y_top'])

def data_x_to_pixel(data_x, bounds):
    return int(bounds['x0'] + data_x * (bounds['x1'] - bounds['x0']))

def is_colored(rgba, target_hex, tol=50):
    tr = int(target_hex[1:3], 16)
    tg = int(target_hex[3:5], 16)
    tb = int(target_hex[5:7], 16)
    r, g, b = int(rgba[0]), int(rgba[1]), int(rgba[2])
    a = int(rgba[3]) if len(rgba) > 3 else 255
    if a < 100:
        return False
    return (abs(r-tr) + abs(g-tg) + abs(b-tb)) < tol

def find_clusters_at_x(x, bounds, arr, hex_color, tol=50):
    """Find vertical clusters of colored pixels at a given x-column.
    Returns list of (y_start, y_end, cluster_height) for each cluster.
    y_start < y_end (both are pixel coordinates, increasing downward)."""
    tr = int(hex_color[1:3], 16)
    tg = int(hex_color[3:5], 16)
    tb = int(hex_color[5:7], 16)
    
    clusters = []
    in_cluster = False
    cs = 0
    for py in range(bounds['y_top'], bounds['y_bottom'] + 1):
        rgba = arr[py, x]
        is_match = (abs(int(rgba[0])-tr)+abs(int(rgba[1])-tg)+abs(int(rgba[2])-tb)) < tol and rgba[3] > 100
        if is_match and not in_cluster:
            cs = py
            in_cluster = True
        elif not is_match and in_cluster:
            clusters.append((cs, py - 1))
            in_cluster = False
    if in_cluster:
        clusters.append((cs, bounds['y_bottom']))
    return clusters

def extract_one_curve(bounds, arr, hex_color, tol=50, debug=False):
    """Extract curve from left subplot (ID) by selecting the correct
    cluster at each x-position. The curve is expected near the diagonal,
    so at low x the y should be low, and it should increase gradually.
    When multiple clusters exist, pick the one that maintains continuity
    with the previous point."""
    results = []
    prev_dy = None
    
    for cl in CONF_LEVELS:
        px = data_x_to_pixel(cl, bounds)
        clusters = find_clusters_at_x(px, bounds, arr, hex_color, tol)
        
        if not clusters:
            results.append(None)
            continue
        
        # Convert clusters to data coordinates
        cluster_data = []
        for cs, ce in clusters:
            py_center = (cs + ce) / 2.0
            dy = pixel_y_to_data(py_center, bounds)
            height = ce - cs + 1
            cluster_data.append((dy, py_center, height, cs, ce))
        
        if len(clusters) == 1:
            best = cluster_data[0]
        else:
            # Multiple clusters: pick the one closest to the previous dy
            # If no previous, pick the one nearest the diagonal (dy ≈ cl)
            if prev_dy is not None:
                best = min(cluster_data, key=lambda cd: abs(cd[0] - prev_dy))
            else:
                # First point: pick cluster nearest the diagonal
                best = min(cluster_data, key=lambda cd: abs(cd[0] - cl))
        
        dy, py_center, height, cs, ce = best
        results.append(dy)
        
        if debug:
            print(f"  cl={cl:.2f}: px={px}, clusters={[(f'{c[0]:.3f}', c[3], c[4], c[2]) for c in cluster_data]}, "
                  f"chosen=dy={dy:.3f} (py={py_center:.1f}), height={height}px")
        
        prev_dy = dy
    
    return results

# ── 3. Extract curves ─────────────────────────────────────────────────────────
METHOD_COLORS = {
    'Laplace':        '#d62728',
    'Bayesian-Torch': '#1f77b4',
}

print("\n=== Left subplot (ID) ===")
id_data = {}
for label, hex_color in METHOD_COLORS.items():
    print(f"\n{label}:")
    observed = extract_one_curve(left_data, arr, hex_color, tol=50, debug=True)
    
    # Fill any missing
    obs = np.array([o if o is not None else np.nan for o in observed])
    valid = ~np.isnan(obs)
    if not np.all(valid):
        if valid.sum() >= 2:
            obs = np.interp(CONF_LEVELS, CONF_LEVELS[valid], obs[valid])
        else:
            obs = np.full_like(obs, 0.5)
    
    id_data[label] = obs
    cal_mae = float(np.mean(np.abs(obs - CONF_LEVELS)))
    print(f"  Cal_MAE = {cal_mae:.4f}")

print("\n=== Right subplot (OOD) ===")
ood_data = {}
for label, hex_color in METHOD_COLORS.items():
    print(f"\n{label}:")
    observed = extract_one_curve(right_data, arr, hex_color, tol=50, debug=False)
    
    obs = np.array([o if o is not None else np.nan for o in observed])
    valid = ~np.isnan(obs)
    if not np.all(valid):
        if valid.sum() >= 2:
            obs = np.interp(CONF_LEVELS, CONF_LEVELS[valid], obs[valid])
        else:
            obs = np.full_like(obs, 0.5)
    
    ood_data[label] = obs
    cal_mae = float(np.mean(np.abs(obs - CONF_LEVELS)))
    print(f"  Cal_MAE = {cal_mae:.4f}")

# ── 4. Replot as merged single-panel ─────────────────────────────────────────
print("\nGenerating single-panel reliability diagram (ID + OOD)...")
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Perfect', zorder=5)

curves = [
    ('Laplace ID',     id_data['Laplace'],        '#d62728', '-',  'o', 0.044),
    ('Laplace OOD',    ood_data['Laplace'],        '#d62728', '--', 'o', 0.029),
    ('Bayesian-Torch ID',  id_data['Bayesian-Torch'],  '#1f77b4', '-',  's', 0.238),
    ('Bayesian-Torch OOD', ood_data['Bayesian-Torch'],  '#1f77b4', '--', 's', 0.281),
]

for name, data, color, ls, marker, cal_mae in curves:
    ax.plot(CONF_LEVELS, data, color=color, lw=1.5, linestyle=ls, marker=marker, ms=4,
            label=f'{name} (Cal_MAE={cal_mae:.3f})')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('Expected confidence')
ax.set_ylabel('Observed coverage')
ax.set_title('PovertyMap — Calibration Reliability')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()

fig_path = f"{OUT}/reliability_povertymap.png"
fig.savefig(fig_path)
print(f"  Saved: {fig_path}")

# ── 5. Diff overlay ──────────────────────────────────────────────────────────
print("\nGenerating diff overlay...")
fig2, ax2 = plt.subplots(figsize=(8, 6))

crop = img.crop((left_data['x0'], left_data['y_top'], left_data['x1']+1, left_data['y_bottom']+1))
ax2.imshow(crop, extent=[0, 1, 0, 1], aspect='auto', alpha=0.6)

diff_curves = [
    ('Laplace ID',     id_data['Laplace'],        '#d62728'),
    ('Laplace OOD',    ood_data['Laplace'],        '#d62728'),
    ('Bayesian-Torch ID',  id_data['Bayesian-Torch'],  '#1f77b4'),
    ('Bayesian-Torch OOD', ood_data['Bayesian-Torch'],  '#1f77b4'),
]
for name, data, color in diff_curves:
    ax2.plot(CONF_LEVELS, data, color=color, lw=3, ms=10,
             label=f'{name} (extracted)', alpha=0.9)
ax2.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect', alpha=0.6)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_xlabel('Expected confidence')
ax2.set_ylabel('Observed coverage')
ax2.set_title('Extraction Diff — extracted curves overlaid on original subplot')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()

diff_path = f"{OUT}/reliability_povertymap_diff.png"
fig2.savefig(diff_path)
print(f"  Saved: {diff_path}")

# ── 6. OOD-only merged panel ────────────────────────────────────────────────
print("\nGenerating OOD-only reliability diagram...")
fig3, ax3 = plt.subplots(figsize=(6, 4.5))
ax3.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Perfect', zorder=5)

ood_curves = [
    ('Laplace OOD',     ood_data['Laplace'],        '#d62728', '-',  'o', 0.029),
    ('Bayesian-Torch OOD', ood_data['Bayesian-Torch'],  '#1f77b4', '-',  's', 0.281),
]
for name, data, color, ls, marker, cal_mae in ood_curves:
    ax3.plot(CONF_LEVELS, data, color=color, lw=1.5, linestyle=ls, marker=marker, ms=5,
             label=f'{name} (Cal_MAE={cal_mae:.3f})')

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.set_xlabel('Expected confidence')
ax3.set_ylabel('Observed coverage')
ax3.set_title('PovertyMap — OOD Calibration Reliability')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
fig3.tight_layout()

ood_path = f"{OUT}/reliability_povertymap_ood.png"
fig3.savefig(ood_path)
print(f"  Saved: {ood_path}")

# ── 7. OOD diff overlay ─────────────────────────────────────────────────────
print("\nGenerating OOD diff overlay...")
fig4, ax4 = plt.subplots(figsize=(8, 6))

crop_ood = img.crop((right_data['x0'], right_data['y_top'], right_data['x1']+1, right_data['y_bottom']+1))
ax4.imshow(crop_ood, extent=[0, 1, 0, 1], aspect='auto', alpha=0.6)

ood_diff_curves = [
    ('Laplace OOD',     ood_data['Laplace'],        '#d62728'),
    ('Bayesian-Torch OOD', ood_data['Bayesian-Torch'],  '#1f77b4'),
]
for name, data, color in ood_diff_curves:
    ax4.plot(CONF_LEVELS, data, color=color, lw=3, ms=10,
             label=f'{name} (extracted)', alpha=0.9)
ax4.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect', alpha=0.6)

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.set_xlabel('Expected confidence')
ax4.set_ylabel('Observed coverage')
ax4.set_title('OOD Extraction Diff — extracted overlaid on original subplot')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
fig4.tight_layout()

ood_diff_path = f"{OUT}/reliability_povertymap_ood_diff.png"
fig4.savefig(ood_diff_path)
print(f"  Saved: {ood_diff_path}")

# ── 8. Report ────────────────────────────────────────────────────────────────
print(f"\n{'Method':<20} {'ID Cal_MAE':<12} {'OOD Cal_MAE':<12}")
print("-"*44)
for label in ['Laplace', 'Bayesian-Torch']:
    id_cal = float(np.mean(np.abs(id_data[label] - CONF_LEVELS)))
    ood_cal = float(np.mean(np.abs(ood_data[label] - CONF_LEVELS)))
    print(f"{label:<20} {id_cal:<12.4f} {ood_cal:<12.4f}")

print("\nDone.")
