def predictive_uncertainties(model, X, n_samples=100):
    """Per-point conf, total, aleatoric, epistemic."""
    samples = model.predictive_samples(X, n_samples=n_samples)
    mean_p = samples.mean(axis=0)
    total = _entropy(mean_p)
    aleatoric = _entropy(samples).mean(axis=0)
    epistemic = total - aleatoric   # mutual information
    conf = mean_p.max(axis=1)
    return conf, total, aleatoric, epistemic

id_conf, _, id_alea, id_epi = predictive_uncertainties(la, X_test)
ood_conf, _, ood_alea, ood_epi = predictive_uncertainties(la, X_ood)
delta_epist = ood_epi.mean() - id_epi.mean()
