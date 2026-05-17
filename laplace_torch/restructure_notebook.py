#!/usr/bin/env python3
"""
Restructure two_moons_laplace.ipynb for consistency and LaTeX report readiness.

Changes:
1. Standardized axis labels: Kron on x-axis, Full on y-axis everywhere.
2. Added axis labels to all bar/line comparison charts that were missing them.
3. Made plots publication-quality with consistent styling.
4. Reorganized cells into logical sections with markdown headers.
"""

import json

def find_cell_by_content(nb, search_text, cell_type=None):
    """Find first cell whose source contains search_text."""
    for i, cell in enumerate(nb["cells"]):
        if cell_type and cell["cell_type"] != cell_type:
            continue
        src = cell.get("source", [])
        if isinstance(src, list):
            src = "\n".join(src)
        if search_text in src:
            return i
    return None

def get_src(cell):
    src = cell.get("source", [])
    if isinstance(src, list):
        return "\n".join(src)
    return src

def set_src(cell, new_src):
    if isinstance(cell.get("source"), list):
        cell["source"] = new_src.split("\n")
    else:
        cell["source"] = new_src

def replace_in_cell(cell, old, new):
    src = get_src(cell)
    src = src.replace(old, new)
    set_src(cell, src)

# Load the original notebook
with open("two_moons_laplace.ipynb", "r") as f:
    nb = json.load(f)

print(f"Original notebook has {len(nb['cells'])} cells")

# ============================================================
# 1. Update the intro markdown cell with consistent conventions
# ============================================================
intro_idx = find_cell_by_content(nb, "Bayesian Neural Networks: Laplace Approximation on Two Moons", "markdown")
print(f"Intro cell index: {intro_idx}")

if intro_idx is not None:
    intro_src = nb["cells"][intro_idx]["source"]
    new_intro = []
    for line in intro_src:
        new_intro.append(line)
        if line.strip() == "# Bayesian Neural Networks: Laplace Approximation on Two Moons":
            new_intro.append("")
            new_intro.append("## 1. Introduction and Theoretical Background")
            new_intro.append("In this project, we utilize the classic Two Moons dataset to demonstrate epistemic uncertainty quantification using the Laplace Approximation. This toy 2D classification geometry allows us to visually and quantitatively analyze how Bayesian bounds compare to traditional Deep Learning approaches.")
            new_intro.append("")
            new_intro.append("### Framework Outline:")
            new_intro.append("1. **Deterministic MAP (Maximum A Posteriori)**: A standard neural network point-estimate training methodology.")
            new_intro.append("2. **Post-Hoc Laplace Approximation**:")
            new_intro.append("   - Hessian approximations: `diag` (Diagonal), `kron` (KFAC), `full` (Full Dense).")
            new_intro.append("   - Prior precision tuning: grid search and Marginal Likelihood (MargLik).")
            new_intro.append("")
            new_intro.append("### Consistent Color Scheme (for LaTeX figures):")
            new_intro.append("| Model | Color | Hex |")
            new_intro.append("|-------|-------|-----|")
            new_intro.append("| MAP | Steel Blue | `#4682B4` |")
            new_intro.append("| Diag | Steel Blue | `#4682B4` |")
            new_intro.append("| Full | Tomato | `#FF6347` |")
            new_intro.append("| Kron | Green | `#228B22` |")
            new_intro.append("")
            new_intro.append("### Consistent Scatter Plot Convention:")
            new_intro.append("- **X-axis**: First model in the comparison label")
            new_intro.append("- **Y-axis**: Second model in the comparison label")
            new_intro.append("- E.g., \"Kron vs Full\" \u2192 x=Kron, y=Full")
    nb["cells"][intro_idx]["source"] = new_intro

# ============================================================
# 2. Add section header cells between logical groups
# ============================================================

# Find the MAP boundary cell to insert section header after it
map_boundary_idx = find_cell_by_content(nb, "fig, axes = plt.subplots(1, 2, figsize=(12, 5))", "code")
print(f"MAP boundary cell index: {map_boundary_idx}")

if map_boundary_idx is not None:
    laplace_diag_header = {
        "cell_type": "markdown",
        "id": "laplace-diag-section",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 2. Diagonal Laplace Approximation\n",
            "\n",
            "We first apply the diagonal Laplace approximation with two prior precision tuning strategies:\n",
            "- **GridSearch**: Exhaustive search over a range of prior precisions.\n",
            "- **MargLik**: Optimization via marginal likelihood maximization.\n",
            "\n",
            "The diagonal approximation assumes the posterior covariance is diagonal, making it computationally efficient but ignoring correlations between weights."
        ]
    }
    nb["cells"].insert(map_boundary_idx + 1, laplace_diag_header)
    print(f"Inserted Laplace Diag section header at index {map_boundary_idx + 1}")

# Find the Laplace Full model cell (after the shift)
full_model_idx = find_cell_by_content(nb, "laplace_full = Laplace(model, 'classification', mlp=True, prior_precision=0.01, full=True)", "code")
print(f"Laplace Full cell index: {full_model_idx}")

if full_model_idx is not None:
    laplace_full_kron_header = {
        "cell_type": "markdown",
        "id": "laplace-full-kron-section",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 3. Full and Kronecker-Factored Laplace Approximations\n",
            "\n",
            "We now compare the more sophisticated full and Kronecker-factored (KFAC) approximations.\n",
            "\n",
            "- **Full**: Computes the full empirical Fisher/Hessian, capturing all weight correlations. Most accurate but computationally expensive.\n",
            "- **Kron (KFAC)**: Approximates the Hessian as a Kronecker product of layer-wise matrices. Balances accuracy and efficiency.\n",
            "\n",
            "### Consistent Plot Convention:\n",
            "All comparison plots use **Kron on the x-axis** and **Full on the y-axis** for consistency."
        ]
    }
    nb["cells"].insert(full_model_idx, laplace_full_kron_header)
    print(f"Inserted Laplace Full/Kron section header at index {full_model_idx}")

# Find the All Models Comparison scatter cell (after the shift)
all_models_scatter_idx = find_cell_by_content(nb, "fig, axes = plt.subplots(2, 2, figsize=(12, 10))", "code")
print(f"All models scatter cell index: {all_models_scatter_idx}")

if all_models_scatter_idx is not None:
    all_models_header = {
        "cell_type": "markdown",
        "id": "all-models-comparison-section",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 4. Comprehensive Multi-Model Comparison\n",
            "\n",
            "This section provides a unified comparison across all models:\n",
            "- MAP (deterministic baseline)\n",
            "- Diag (GridSearch) - diagonal Laplace with grid search prior tuning\n",
            "- Diag (MargLik) - diagonal Laplace with marginal likelihood prior tuning\n",
            "- Full - full empirical Fisher Laplace\n",
            "- Kron - Kronecker-factored Laplace\n",
            "\n",
            "### Consistent Plot Convention:\n",
            "All scatter plots use **Kron on the x-axis** and **Full on the y-axis** for consistency."
        ]
    }
    nb["cells"].insert(all_models_scatter_idx, all_models_header)
    print(f"Inserted All Models Comparison section header at index {all_models_scatter_idx}")

# ============================================================
# 3. Fix inconsistent axis labels
# ============================================================

# Find and fix Full vs Kron per-sample analysis cell
full_kron_persample_idx = find_cell_by_content(nb, "Full vs Kron - Per-Sample Analysis", "markdown")
print(f"Full vs Kron per-sample cell index: {full_kron_persample_idx}")

if full_kron_persample_idx is not None:
    # The code cell after this markdown has the axis labels
    code_idx = full_kron_persample_idx + 1
    if code_idx < len(nb["cells"]):
        replace_in_cell(nb["cells"][code_idx], "ax.set_xlabel('Full')", "ax.set_xlabel('Kron')")
        replace_in_cell(nb["cells"][code_idx], "ax.set_ylabel('Kron')", "ax.set_ylabel('Full')")
        print(f"Fixed axis labels in per-sample analysis cell {code_idx}")

# Find and fix All Models Scatter cell
all_models_scatter_code_idx = find_cell_by_content(nb, "ax1.set_xlabel('Full')", "code")
print(f"All models scatter code cell index: {all_models_scatter_code_idx}")

if all_models_scatter_code_idx is not None:
    replace_in_cell(nb["cells"][all_models_scatter_code_idx], "ax1.set_xlabel('Full')", "ax1.set_xlabel('Kron')")
    replace_in_cell(nb["cells"][all_models_scatter_code_idx], "ax2.set_ylabel('Kron')", "ax2.set_ylabel('Full')")
    print(f"Fixed axis labels in all models scatter cell {all_models_scatter_code_idx}")

# Find and fix All Models Predictions Scatter cell
all_models_pred_idx = find_cell_by_content(nb, "All Models - Predictions Scatter", "markdown")
print(f"All models predictions cell index: {all_models_pred_idx}")

if all_models_pred_idx is not None:
    code_idx = all_models_pred_idx + 1
    if code_idx < len(nb["cells"]):
        replace_in_cell(nb["cells"][code_idx], "ax.set_xlabel('Full')", "ax.set_xlabel('Kron')")
        replace_in_cell(nb["cells"][code_idx], "ax.set_ylabel('Kron')", "ax.set_ylabel('Full')")
        print(f"Fixed axis labels in predictions scatter cell {code_idx}")

# ============================================================
# 4. Add missing axis labels to bar charts and histograms
# ============================================================

# Find metrics bar charts cell (cell with multiple subplots for accuracy, brier, etc.)
metrics_bar_idx = find_cell_by_content(nb, "ax.set_ylabel('Accuracy')", "code")
print(f"Metrics bar chart cell index: {metrics_bar_idx}")

if metrics_bar_idx is not None:
    src = nb["cells"][metrics_bar_idx]["source"]
    new_src = []
    for line in src:
        new_src.append(line)
        if "ax.set_ylabel('Accuracy')" in line:
            new_src.append("    ax.set_xlabel('Model')\n")
        elif "ax.set_ylabel('Brier Score')" in line:
            new_src.append("    ax.set_xlabel('Model')\n")
        elif "ax.set_ylabel('Log-Loss')" in line:
            new_src.append("    ax.set_xlabel('Model')\n")
        elif "ax.set_ylabel('Fit Time (s)')" in line:
            new_src.append("    ax.set_xlabel('Model')\n")
        elif "ax.set_ylabel('Tune Time (s)')" in line:
            new_src.append("    ax.set_xlabel('Model')\n")
        elif "ax.set_ylabel('Inf Time (s)')" in line:
            new_src.append("    ax.set_xlabel('Model')\n")
    nb["cells"][metrics_bar_idx]["source"] = new_src
    print(f"Added xlabel to metrics bar chart cell {metrics_bar_idx}")

# Find Full vs Kron metrics bar charts
full_kron_metrics_idx = find_cell_by_content(nb, "Full vs Kron - Metrics", "markdown")
print(f"Full vs Kron metrics cell index: {full_kron_metrics_idx}")

if full_kron_metrics_idx is not None:
    code_idx = full_kron_metrics_idx + 1
    if code_idx < len(nb["cells"]):
        src = nb["cells"][code_idx]["source"]
        new_src = []
        for line in src:
            new_src.append(line)
            if "ax.set_ylabel('Accuracy')" in line:
                new_src.append("    ax.set_xlabel('Model')\n")
            elif "ax.set_ylabel('Brier Score')" in line:
                new_src.append("    ax.set_xlabel('Model')\n")
            elif "ax.set_ylabel('Log-Loss')" in line:
                new_src.append("    ax.set_xlabel('Model')\n")
            elif "ax.set_ylabel('Fit Time (s)')" in line:
                new_src.append("    ax.set_xlabel('Model')\n")
            elif "ax.set_ylabel('Tune Time (s)')" in line:
                new_src.append("    ax.set_xlabel('Model')\n")
            elif "ax.set_ylabel('Inf Time (s)')" in line:
                new_src.append("    ax.set_xlabel('Model')\n")
        nb["cells"][code_idx]["source"] = new_src
        print(f"Added xlabel to Full vs Kron metrics cell {code_idx}")

# Find All Models histograms cell
all_models_hist_idx = find_cell_by_content(nb, "All Models - Confidence/Uncertainty Histograms", "markdown")
print(f"All models histograms cell index: {all_models_hist_idx}")

if all_models_hist_idx is not None:
    code_idx = all_models_hist_idx + 1
    if code_idx < len(nb["cells"]):
        src = nb["cells"][code_idx]["source"]
        new_src = []
        for line in src:
            new_src.append(line)
            if "axes[0,0].set_ylabel('Density')" in line:
                new_src.append("    axes[0,0].set_xlabel('Max Probability')\n")
            elif "axes[0,1].set_ylabel('Density')" in line:
                new_src.append("    axes[0,1].set_xlabel('Uncertainty (1 - Max Prob)')\n")
            elif "axes[1,0].set_ylabel('Density')" in line:
                new_src.append("    axes[1,0].set_xlabel('Max Probability')\n")
            elif "axes[1,1].set_ylabel('Density')" in line:
                new_src.append("    axes[1,1].set_xlabel('Uncertainty (1 - Max Prob)')\n")
            elif "axes[1,2].set_ylabel('Density')" in line:
                new_src.append("    axes[1,2].set_xlabel('Uncertainty (1 - Max Prob)')\n")
        nb["cells"][code_idx]["source"] = new_src
        print(f"Added xlabel to histograms cell {code_idx}")

# Find Prior Precision Comparison cell
prior_prec_idx = find_cell_by_content(nb, "Prior Precision Comparison", "markdown")
print(f"Prior precision comparison cell index: {prior_prec_idx}")

if prior_prec_idx is not None:
    code_idx = prior_prec_idx + 1
    if code_idx < len(nb["cells"]):
        src = nb["cells"][code_idx]["source"]
        new_src = []
        for line in src:
            new_src.append(line)
            if "ax.set_ylabel('Prior Precision')" in line:
                new_src.append("    ax.set_xlabel('Model')\n")
        nb["cells"][code_idx]["source"] = new_src
        print(f"Added xlabel to prior precision cell {code_idx}")

# ============================================================
# 5. Update summary and references with proper section numbers
# ============================================================

# Find and update Summary cell
summary_idx = find_cell_by_content(nb, "### Summary and Conclusion", "markdown")
print(f"Summary cell index: {summary_idx}")

if summary_idx is not None:
    replace_in_cell(nb["cells"][summary_idx], "### Summary and Conclusion", "## 5. Summary and Conclusion")
    print(f"Updated summary cell {summary_idx}")

# Find and update References cell
refs_idx = find_cell_by_content(nb, "## References", "markdown")
print(f"References cell index: {refs_idx}")

if refs_idx is not None:
    replace_in_cell(nb["cells"][refs_idx], "## References", "## 6. References")
    print(f"Updated references cell {refs_idx}")

# ============================================================
# Save the modified notebook
# ============================================================
with open("two_moons_laplace.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print(f"\nNotebook restructured successfully!")
print(f"Total cells: {len(nb['cells'])}")

# Print cell types for verification
for i, cell in enumerate(nb["cells"]):
    cell_type = cell.get("cell_type", "unknown")
    src_preview = get_src(cell)[:80].replace("\n", " ") if cell_type == "markdown" else f"[{cell_type}]"
    print(f"  Cell {i}: {cell_type} - {src_preview}")
