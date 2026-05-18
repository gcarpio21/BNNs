import nbformat as nbf
import json

with open('laplace_torch/two_moons_bayesian.ipynb', 'r') as f:
    nb = nbf.read(f, as_version=4)

# ── Add Cell for distributions per sample ──
new_cell = nbf.v4.new_code_cell("""# ─── Compute per-sample predictions ───
def get_confidence_and_uncertainty(model, n_samples=30):
    model.eval()
    probs_list = []
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        for _ in range(n_samples):
            logits = model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_list.append(probs)
    
    probs_stack = np.stack(probs_list)
    mean_probs = np.mean(probs_stack, axis=0) # shape: (len(X_test), 2)
    conf = mean_probs.max(axis=1) # Max probability
    unc = 1 - conf # Epistemic representation mapped from prob margin
    return conf, unc

flip_conf, flip_unc = get_confidence_and_uncertainty(model_flipout)
bbb_conf, bbb_unc = get_confidence_and_uncertainty(model_bbb)

# ─── Figure: Analysis Distributions ───
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
COLOR_FLIP = '#FF6347' # tomato
COLOR_BBB  = '#228B22' # green

# 1. Confidence distributions
axes[0].hist(flip_conf, bins=40, alpha=0.5, label='Flipout', color=COLOR_FLIP, density=True)
axes[0].hist(bbb_conf, bins=40, alpha=0.5, label='BBB/Reparam', color=COLOR_BBB, density=True)
axes[0].set_xlabel('Max probability')
axes[0].set_title('Confidence distributions')
axes[0].legend()

# 2. Scatter Map
def make_scatter_comparison(ax, x_label, x_data, y_label, y_data, title, c_data=y_test):
    ax.scatter(x_data, y_data, alpha=0.4, s=15, c=c_data, cmap='bwr', edgecolors='k', linewidths=0.2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlim([0.45, 1.05])
    ax.set_ylim([0.45, 1.05])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

make_scatter_comparison(axes[1], 'Flipout confidence', flip_conf,
                        'BBB confidence', bbb_conf,
                        'Flipout vs BBB — per sample')

# 3. Uncertainty distributions
axes[2].hist(flip_unc, bins=40, alpha=0.5, label='Flipout', color=COLOR_FLIP, density=True)
axes[2].hist(bbb_unc, bins=40, alpha=0.5, label='BBB/Reparam', color=COLOR_BBB, density=True)
axes[2].set_xlabel('Uncertainty')
axes[2].set_title('Uncertainty distributions')
axes[2].legend()

plt.suptitle('Flipout vs BBB — Confidence and Uncertainty analysis', fontsize=13)
plt.tight_layout()
plt.savefig('../figures/bayesian_torch_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
""")

nb.cells.append(new_cell)

with open('laplace_torch/two_moons_bayesian.ipynb', 'w') as f:
    nbf.write(nb, f)

