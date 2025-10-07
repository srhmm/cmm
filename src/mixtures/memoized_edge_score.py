from typing import List, Tuple

import numpy as np

from src.mixtures.mixing.mixing import conditional_mixture_known_assgn, MixingType
from src.mixtures.mixing.mixing import fit_conditional_mixture
from src.mixtures.mixing.regression import DataType, ScoreType
from src.mixtures.mixing.regression import fit_fun, RegressorType
from src.mixtures.mixing.scoring import compute_edge_score
from src.mixtures.util.tc_discrete import total_correlation
from src.mixtures.util.utils_idl import _get_true_idl, _get_true_idl_Z


class MemoizedEdgeScore:
    score_type: ScoreType

    def __init__(self, mixing_type, X, **scoring_parameters):
        self.X = X
        self.mixing_type = mixing_type
        self.defaultargs = {
            #"data_type": DataType.CONTINUOUS,
            "score_type": ScoreType.GAM,
            "hybrid": False, # "is_true_edge": lambda i: lambda j: "",
            "k_max": 5,
            "oracle_Z": False,
            "oracle_K": False,
            "lambda_mix": 1,
            "known_true_order": False, "true_idls": None, "t_A": None, "t_Z": None, "t_n_Z": None, "true_top_order": [],
            "lg": None, "vb": 0}

        assert all([arg in self.defaultargs.keys() for arg in scoring_parameters.keys()])
        self.__dict__.update((k, v) for k, v in self.defaultargs.items() if k not in scoring_parameters.keys())
        self.__dict__.update((k, v) for k, v in scoring_parameters.items() if k in self.defaultargs.keys())
        self._info = lambda st: (self.lg.info(st) if self.lg is not None else print(st)) if self.vb > 0 else None

        # Memoized info
        self.score_cache = {}
        self.idl_cache = {}
        self.pproba_cache = {}
        self.idl_dicts = {}
        self.resid_cache = {}

        # params needed for scoring edges
        self.scre_params = dict(score_type=self.score_type, lambda_mix=self.lambda_mix, vb=self.vb, lg=self.lg)
        self.idl_params = dict(
            score_type=self.score_type,
            k_max=self.k_max,
            oracle_Z=self.oracle_Z,
            oracle_K=self.oracle_K,
            true_idls=self.true_idls,
            t_A=self.t_A,
            t_Z=self.t_Z,
            t_n_Z=self.t_n_Z,
            vb=self.vb, lg=self.lg
        )

    def score_edge(self, j, pa, idl_dict=dict()) -> int:
        """
        Evaluates score for a causal relationship pa(Xj)->Xj.

        :param j: Xj
        :param pa: pa(Xj)
        :return: score_up=score(Xpa->Xj)
        """
        hash_key = f'j_{str(j)}_pa_{str(pa)}'

        if self.score_cache.__contains__(hash_key):
            return self.score_cache[hash_key]
        if self.hybrid: assert idl_dict.__contains__('idl')

        score = compute_edge_score(
            self.X, covariates=pa, target=j, idl_dict=idl_dict, **self.scre_params)

        self.score_cache[hash_key] = score
        return score

    def idl_edge(self, j, pa) -> [List, List, dict]:
        """
        Gets causal clustering for relationship pa(Xj)->Xj.

        :param j: Xj
        :param pa: pa(Xj)
        :return: hard clust, post proba, dict
        """
        hash_key = f'j_{str(j)}_pa_{str(pa)}'

        if self.idl_cache.__contains__(hash_key) and self.pproba_cache.__contains__(hash_key):
            return self.idl_cache[hash_key], self.pproba_cache[hash_key], self.idl_dicts[hash_key]

        resi = None if self.idl_params["oracle_Z"] else self.resid_edge(j, pa)
        idl, pproba, idl_dict = idl_and_latent_bic_score(self.mixing_type,
                                                         self.X, covariates=pa, target=j, resid=resi, **self.idl_params)
        self.idl_cache[hash_key], self.pproba_cache[hash_key], self.idl_dicts[hash_key] = idl, pproba, idl_dict
        return idl, pproba, idl_dict


    def resid_edge(self, j: int, pa: [int]) -> np.array:
        """Residual for average functional model for edge pa to node j."""
        hash_key = f"j_{str(j)}_pa_{str(pa)}"

        if self.resid_cache.__contains__(hash_key):
            return self.resid_cache[hash_key]

        resids, strength = fit_fun(self.X[:, pa], self.X[:, j], RegressorType.LN, 42)

        self.resid_cache[hash_key] = resids
        return resids



def local_score_latent_bic(
    Data: Tuple[np.ndarray, int], i: int, PAi: List[int], parameters=None
) -> float:
    """ compute the latent-aware BIC, to do so we fit a MLR using EM """


    if parameters is None: kmax = 5
    else: kmax = parameters.get("k_max", 5)
    params = {"k_max" : kmax, "oracle_K": False, "oracle_Z": False}

    idl, pproba, result_dict = idl_and_latent_bic_score(MixingType.MIX_LIN, Data, PAi, i, None, **params)
    return int(result_dict["bic"])


def idl_and_latent_bic_score(
        mixing_type,
        X,
        covariates: list,
        target: int,
        resid=None,
        **params) -> [List, List, dict]:
    if params.get("true_idls") is not None:
        true_idl = _get_true_idl(params["true_idls"], covariates, target, params["t_A"])
    elif params.get("t_Z") is not None:
        true_idl = _get_true_idl_Z(
            covariates, target, params["t_A"], params["t_Z"], params["t_n_Z"], X.shape[0])
    else: true_idl = None

    if params["oracle_Z"]:
        assert true_idl is not None
        true_idl, true_pproba, true_dict = conditional_mixture_known_assgn(
            X=X, node_i=target, pa_i=covariates, true_idl=true_idl, **params)
        return true_idl, true_pproba, true_dict

    range_k = range(1, params["k_max"] + 1) if not params["oracle_K"] else [len(np.unique(true_idl))]
    estim_idl, estim_post_proba, idl_dict = fit_conditional_mixture(
        mty=mixing_type, X=X, node_i=target, pa_i=covariates, range_k=range_k, resid=resid, true_idl=true_idl,
        lg=params.get("lg", None))
    return estim_idl, estim_post_proba, idl_dict
