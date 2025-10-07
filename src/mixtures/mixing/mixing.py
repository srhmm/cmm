from enum import Enum

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, HDBSCAN
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score
from sklearn.mixture import GaussianMixture

from src.mixtures.mixing.util import to_params, \
    mix_regression_bic, mix_regression_params_kn_assgn


class MixingType(Enum):
    # mixtures of regressions
    MIX_LIN = 'mixLin'
    # simple baselines
    _BASE_BEST = 'clusBest'  # this selects per node the best MM which is not really fair; select for the whole graph the best one
    BASE_GMM = 'clusGMM'
    BASE_KMEANS = 'clusKmeans'
    BASE_SPECTRAL = 'clusSpectral'
    BASE_DBSCAN = 'clusDBSCAN'
    BASE_HDBSCAN = 'clusHDBSCAN'
    BASE_GMM_GLOB = 'clusGMMglobal'
    BASE_RANDOM_SPLIT = 'clusRandSplit'

    def __eq__(self, other): return self.value == other.value
    def __str__(self): return str(self.value)
    def search_each_node(self): return not self.value.endswith('global')

    def is_unconditional_mixture(self): return self.value.startswith('clus')


def fit_conditional_mixture(mty: MixingType, **kwargs):
    if mty.value.startswith('mix'):
        return fit_functional_mixture(**kwargs)
    elif mty.value.startswith('resid'):
        return fit_resid_mixture(mty, **kwargs)
    elif mty.value.startswith('clus'):
        return fit_marginal_mixture(mty, **kwargs)
    else:
        raise ValueError(mty)


def _fit_best_mixture(X, range_k, true_idl, sim_score=adjusted_mutual_info_score, sim_min=-np.inf):
    best_ami = sim_min
    best_arg = None
    for mty in [MixingType.BASE_GMM, MixingType.BASE_KMEANS, MixingType.BASE_SPECTRAL, MixingType.BASE_DBSCAN]:
        idl, pproba, div = fit_mixture_model(mty, X, range_k, None)
        ami = sim_score(true_idl, idl)
        if ami > best_ami:
            best_ami = ami
            best_arg = idl, pproba, div
    return best_arg[0], best_arg[1], best_arg[2]


def fit_mixture_model(mty, X, range_k, true_idl=None, kchoice_score=silhouette_score, kchoice_threshold=0.5,
                      kchoice_min=-1):
    if mty == MixingType.BASE_RANDOM_SPLIT:
        assert true_idl is not None
        true_k = len(np.unique(true_idl))
        # sample random labels with true k
        rand_split = np.random.choice(true_k, size=len(true_idl))
        return rand_split, None, dict()

    elif mty == MixingType._BASE_BEST:
        assert true_idl is not None
        return _fit_best_mixture(X, range_k, true_idl)

    elif mty in [MixingType.BASE_GMM, MixingType.BASE_GMM_GLOB]:
        mm = GaussianMixture
        best_bic, best_k, best_m = np.inf, 0, None
        for k in range_k:
            gm = mm(k)
            gm.fit(X)
            bic_k = gm.bic(X)
            if bic_k < best_bic: best_bic, best_k, best_m = bic_k, k, gm

        return best_m.predict(X), best_m.predict_proba(X), dict(bic=best_bic,
                                                                idl=best_m.predict(X), pproba=best_m.predict_proba(X))
    elif mty == MixingType.BASE_DBSCAN:
        mm = DBSCAN().fit(X)
        return mm.labels_, None, dict()
    elif mty == MixingType.BASE_HDBSCAN:
        mm = HDBSCAN().fit(X)
        return mm.labels_, None, dict()
    else:
        model = KMeans if mty == MixingType.BASE_KMEANS \
            else SpectralClustering if mty == MixingType.BASE_SPECTRAL else None
        if model is None: raise ValueError(mty)
        best_s, best_k, best_idl = kchoice_min, 1, None
        for k in range_k:
            if k == 1: continue
            mm = model(n_clusters=k, random_state=42)
            idl = mm.fit_predict(X)
            s = kchoice_score(X, idl)
            if s > best_s: best_s, best_k, best_idl = s, k, idl
        if best_s < kchoice_threshold:  best_idl = model(n_clusters=1, random_state=42).fit_predict(X)

        return best_idl, None, dict()


def fit_marginal_mixture(mty, X, node_i, pa_i, range_k, resid, true_idl, **kwargs):
    X = np.hstack([X[:, pa_i], X[:, node_i].reshape(-1, 1)]) if len(pa_i) > 0 else X[:, node_i].reshape(-1, 1)
    return fit_mixture_model(mty, X, range_k, true_idl)


def fit_resid_mixture(mty, X, node_i, pa_i, range_k, resid, true_idl):
    return fit_mixture_model(mty, resid, range_k)


def fit_functional_mixture(X, node_i, pa_i, range_k, resid, true_idl, lg=None, vb=0):
    if not len(pa_i):
        fit_marginal_mixture(MixingType.BASE_GMM, X, node_i, pa_i, range_k, resid, true_idl)

    # base = importr('base')
    # utils = importr('utils')
    # utils.chooseCRANmirror(ind=1)
    # utils.install_packages('ggplot2')
    # utils.install_packages('lazyeval')
    # utils.install_packages('flexmix')
    # utils.install_packages('stats')
    # utils.install_packages('mvtnorm')
    import rpy2.robjects as robjects
    import numpy as np
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr

    base = importr('base')
    utils = importr('utils')
    flexmix = importr('flexmix')
    numpy2ri.activate()

    data_pa = X[:, pa_i] if len(pa_i) > 0 else np.random.normal(size=X[:, node_i].reshape(-1, 1).shape)
    data_np = np.hstack([X[:, node_i].reshape(-1, 1), data_pa])
    data_r = robjects.r.matrix(data_np, nrow=data_np.shape[0], ncol=data_np.shape[1])
    robjects.r.assign("data_r", data_r)

    r_df = robjects.r['data.frame'](x=data_r)
    linear_formula = f"x.1 ~ " + ' + '.join([f'x.{i + 2}' for i in range(data_pa.shape[1])])
    formula = robjects.Formula(linear_formula)
    best_bic, best_k, best_m = np.inf, 0, None
    bics = []
    for k in range_k:
        m1 = flexmix.flexmix(formula, data=r_df, k=k)
        # print(robjects.r['summary'](m1))
        cluster_probs = robjects.r['posterior'](m1)
        params = robjects.r['parameters'](m1)
        bic = robjects.r['BIC'](m1)
        bics.append(bic)
        params_k = to_params(params)
        # avg_kl, pairwise_kl = mix_regression_separation(params_k, data_np)
        if bic < best_bic:
            best_bic, best_k, best_m = bic, k, m1

        if vb: print(f"\tk={k}, {bic}")  # sepa: {avg_kl}")

    best_params = to_params(robjects.r['parameters'](best_m))

    post_probs = np.array(robjects.r['posterior'](best_m))

    def post_entropy(p_proba, eps=1e-12):
        p_safe = np.clip(p_proba, eps, 1.0)
        return -np.sum(p_safe * np.log(p_safe), axis=1)

    ent_idl = post_entropy(post_probs.reshape(1, -1))
    hard_splt = np.argmax(post_probs, axis=1)

    idl_dict = dict(
        bic=best_bic,
        idl=hard_splt,
        pproba=post_probs,
        entropy=ent_idl
    )

    return hard_splt, post_probs, idl_dict


def conditional_mixture_known_assgn(X, node_i, pa_i, true_idl, **scoring_params):
    """ fit regresssions for a known mix assignment, pproba from log liks of those regressions (todo or degen?) """
    if len(pa_i) > 0:
        (Xx, y) = (X[:, pa_i], X[:, node_i])
        beta_l, sig_l = mix_regression_params_kn_assgn(Xx, y, true_idl)
        bic = mix_regression_bic(Xx, y, true_idl, beta_l, sig_l)
        pproba = None
        ent_idl = 0
    else:
        pproba = None
        bic = 0
        ent_idl = 0
    idl_dict = dict(
        bic=bic,
        idl=true_idl,
        pproba=pproba,
        ent_idl=ent_idl
    )
    return true_idl, pproba, idl_dict
