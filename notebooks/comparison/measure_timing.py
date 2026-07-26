"""
Measures MAP training time and inference time missing from comparison notebooks.
Outputs numbers suitable for the poster timing table.
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parent
while ROOT != ROOT.parent and not (ROOT / "shared").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
from shared import TinyMLP, load_checkpoint, seed_everything, load_two_moons, load_sinusoid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

SEED   = 711
SIN_CKPT = str(ROOT / "results" / "checkpoints" / "sinusoid_comparison_1seed")
TM_CKPT  = str(ROOT / "results" / "checkpoints" / "two_moons_laplace_5seed")
N_REPEATS = 5  # average over repeats for stability

# ── Sinusoid MAP training time ─────────────────────────────────────────────────
print("=" * 55)
print("SINUSOID — MAP training time")
print("=" * 55)

sin_data = load_sinusoid(n_data=600, sigma_noise=0.3, batch_size=64, seed=SEED,
                         device=device, n_val=200, n_test_id=200, n_test_ood=200)
train_loader = sin_data["train_loader"]
X_test_id    = sin_data["X_test_id"]
X_test_ood   = sin_data["X_test_ood"]

def get_sin_net():
    seed_everything(SEED)
    return nn.Sequential(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 1)).to(device)

map_times = []
for _ in range(N_REPEATS):
    net = get_sin_net()
    opt = optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    t0 = time.perf_counter()
    for epoch in range(500):
        net.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt.zero_grad()
            loss = criterion(net(X_b), y_b)
            loss.backward()
            opt.step()
    map_times.append(time.perf_counter() - t0)

print(f"MAP train (for Laplace overhead): {np.mean(map_times):.1f} ± {np.std(map_times):.1f} s  (n={N_REPEATS})")

# ── Sinusoid BNN inference time ────────────────────────────────────────────────
print("\nSINUSOID — BNN inference time (Flipout/scratch, BBB/scratch)")

def load_sin_bnn(fname, model_type):
    net = get_sin_net()
    dnn_to_bnn(net, {"prior_mu": 0.0, "prior_sigma": 1.0,
                     "posterior_mu_init": 0.0, "posterior_rho_init": -3.0,
                     "type": model_type, "moped_enable": False, "moped_delta": 0.5})
    ck = load_checkpoint(f"{SIN_CKPT}/{fname}", map_location=device)
    net.load_state_dict(ck["model_state_dict"])
    return net.to(device).eval()

def time_bnn_inference(net, X, n_mc=100, repeats=10):
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            t0 = time.perf_counter()
            torch.stack([net(X) for _ in range(n_mc)])
            times.append(time.perf_counter() - t0)
    return np.mean(times), np.std(times)

for fname, mtype, label in [
    (f"sinusoid_flipout_scratch_seed{SEED}.pt", "Flipout",            "Flipout / scratch"),
    (f"sinusoid_bbb_scratch_seed{SEED}.pt",     "Reparameterization", "BBB / scratch"),
]:
    net = load_sin_bnn(fname, mtype)
    mu, sd = time_bnn_inference(net, X_test_id.to(device))
    print(f"  {label}: {mu:.3f} ± {sd:.3f} s (ID, n_mc=100, n={10})")

# ── Two Moons MAP training time ────────────────────────────────────────────────
print("\n" + "=" * 55)
print("TWO MOONS — MAP training time (5 seeds)")
print("=" * 55)

SEEDS = [42, 711, 123, 456, 789]
tm_map_times = []
for seed in SEEDS:
    tm_data = load_two_moons(seed=seed, noise=0.3, batch_train=32, batch_eval=64, device=device)
    tl = tm_data["train_loader"]
    net = TinyMLP(hidden=64).to(device)
    opt = optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    t0 = time.perf_counter()
    for epoch in range(200):
        net.train()
        for X_b, y_b in tl:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt.zero_grad()
            loss = criterion(net(X_b), y_b)
            loss.backward()
            opt.step()
    tm_map_times.append(time.perf_counter() - t0)
    print(f"  seed {seed}: {tm_map_times[-1]:.2f} s")

print(f"MAP train mean: {np.mean(tm_map_times):.1f} ± {np.std(tm_map_times):.1f} s")

print("\nDone.")
