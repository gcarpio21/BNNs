import nbformat as nbf
import json

with open('laplace_torch/two_moons_bayesian.ipynb', 'r') as f:
    nb = nbf.read(f, as_version=4)

# ── Add Cell for Metrics and LaTeX export ──
new_cell = nbf.v4.new_code_cell("""# ─── Compute Point Metrics (Accuracy, NLL, Brier) ───
import pandas as pd
import time

def evaluate_metrics(model, model_name, n_samples=30):
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_test, dtype=torch.long).to(device)
    
    inf_start = time.time()
    probs_list = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_list.append(probs)
    inf_time = time.time() - inf_start
    
    # Average probabilities over samples
    mean_probs = np.mean(np.stack(probs_list), axis=0) # shape (N, 2)
    preds = mean_probs.argmax(axis=1)
    
    accuracy = (preds == y_test).mean()
    
    # Negative Log Likelihood
    nll = -np.log(mean_probs[np.arange(len(y_test)), y_test] + 1e-12).mean()
    
    # Brier Score 
    # (mean squared difference between predicted probability and actual outcome)
    y_true_onehot = np.zeros_like(mean_probs)
    y_true_onehot[np.arange(len(y_test)), y_test] = 1
    brier_score = np.mean(np.sum((mean_probs - y_true_onehot)**2, axis=1))

    return {
        "Accuracy": accuracy,
        "NLL": nll,
        "Brier Score": brier_score,
        "Inference Time (s)": inf_time
    }

metrics_dict = {}
metrics_dict["Bayesian-Torch (Flipout)"] = evaluate_metrics(model_flipout, "Flipout")
metrics_dict["Bayesian-Torch (BBB)"] = evaluate_metrics(model_bbb, "BBB")

df_metrics = pd.DataFrame(metrics_dict).T
df_metrics.index.name = "Model Type"

# Format display
df_display = df_metrics.copy()
for col in df_display.columns:
    df_display[col] = df_display[col].map(lambda x: f"{x:.4f}")

print("=== Two Moons Bayesian-Torch Benchmarks ===")
display(df_display)

# Save to CSV
csv_path = "twomoons_bayesian_torch_metrics.csv"
df_metrics.to_csv(csv_path)
print(f"\\nSaved metrics to {csv_path}")

# Output as LaTeX for academic report
print("\\n=== LaTeX Table Code ===")
latex_str = df_display.style.format(precision=4).to_latex()
print(latex_str)

# ── Figure: Metrics table ──
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')
the_table = ax.table(cellText=df_display.values, colLabels=df_display.columns,
                     rowLabels=df_display.index, cellLoc='center', loc='center')
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)
the_table.scale(1.2, 1.8)
plt.savefig('../figures/bayesian_torch_metrics_table.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: figures/bayesian_torch_metrics_table.png')
""")

nb.cells.append(new_cell)

with open('laplace_torch/two_moons_bayesian.ipynb', 'w') as f:
    nbf.write(nb, f)

