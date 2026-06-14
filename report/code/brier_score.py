def brier_score(probs, labels):
    """Mean Brier score for multiclass probs."""
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - one_hot) ** 2).mean())
