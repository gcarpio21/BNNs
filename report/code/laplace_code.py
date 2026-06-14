la = Laplace(model, 'classification',
             subset_of_weights='last_layer',
             hessian_structure='kron')
la.fit(train_loader)
la.optimize_prior_precision(
    method='gridsearch',
    val_loader=val_loader,
    pred_type='glm',
    link_approx='probit')
probs = la(X_test, pred_type='glm', link_approx='probit')
