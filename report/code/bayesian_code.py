dnn_to_bnn(net, {
    "prior_mu": 0.0, "prior_sigma": 1.0,
    "posterior_mu_init": 0.0, "posterior_rho_init": -3.0,
    "type": "Reparameterization",  # or "Flipout"
    "moped_enable": True, "moped_delta": 0.5,
})

# ELBO step
loss = criterion(net(inputs), labels) + get_kl_loss(net) / N

# AvUC step (M=5 MC samples)
logits_mc = torch.stack([net(inputs) for _ in range(5)])
probs = torch.softmax(logits_mc.mean(dim=0), dim=1)
ent = -(probs * torch.log(probs + 1e-12)).sum(dim=1)
avu = AvULoss()(logits_mc.mean(dim=0), labels, ent.mean())
loss = CE + get_kl_loss(net) / N + avu
