import numpy as np

from src.mixtures.mixing.regression import fit_functional_model


def compute_edge_score_idl(
        idl,
        X,
        covariates: list,
        target: int,
        **scoring_params) -> int:
    return sum([compute_edge_score(X[idl == k], covariates, target, **scoring_params) for k in np.unique(idl)])


def compute_edge_score(
        X,
        covariates: list,
        target: int,
        idl_dict=None,
        **scoring_params) -> int:
    """ Computes the score for target node given parents.

    :param X: data
    :param covariates: parents of target
    :param target: node
    :return: edge score
    """
    if 'idl' in idl_dict:
        score = idl_dict["bic"]
    else:
        score = fit_score_functional_model_continuous(
            X, covariates, target, **scoring_params)
    return score


def fit_score_functional_model_continuous(X, covariates, target, **scoring_params):
    if isinstance(X, dict):
        M = X[0].shape[1]
        X_parents = {0: np.random.normal(size=X[0][:, [target]].shape)} if len(covariates) == 0 else {
            0: X[0][:, covariates]}
        X_target = {0: X[0][:, target]}
    else:
        M = X.shape[1]
        X_parents = {0: np.random.normal(size=X[:, [target]].shape)} if len(covariates) == 0 else {
            0: X[:, covariates]}
        X_target = {0: X[:, target]}
    return fit_functional_model(X_parents, X_target, M, **scoring_params)
