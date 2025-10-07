from enum import Enum

import numpy as np
from pygam import GAM
from pygam import LinearGAM
from scipy.interpolate import make_lsq_spline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from src.mixtures.util.util import data_scale


class GPType(Enum):
    EXACT = 'gp'
    FOURIER = 'ff'


class MIType(Enum):
    TC = 'tc'
    MSS = 'mss'


class ScoreType(Enum):
    GAM = 'gam'
    SPLINE = 'spline'
    GP = GPType
    MI = MIType

    def get_model(self, **kwargs):
        raise ValueError(f"Only valid for ScoreType.GP, not {self.value}")

    def is_mi(self):
        return self.value in MIType._member_map_


class DataType(Enum):
    CONTINUOUS = 'cont'
    TIMESERIES = 'time'
    TIME_MCONTEXT = 'time_iv'
    CONT_MCONTEXT = 'cont_iv'
    MIXING = 'mixing'

    def __eq__(self, other):
        return self.value == other.value

    def is_time(self):
        return self.value in [self.TIMESERIES.value, self.TIME_MCONTEXT.value]

    def is_multicontext(self):
        return self.value in [self.TIME_MCONTEXT.value, self.CONT_MCONTEXT.value]


def fit_functional_model(
        X, y, M, **scoring_params):
    r""" fitting and scoring functional models

    :param X: parents
    :param y: target
    :param M: n all nodes
    :param scoring_params: hyperparameters

    :Keyword Arguments:
    * *score_type* (``ScoreType``) -- regressor and associated information-theoretic score
    """
    params_score_type = scoring_params.get("score_type", ScoreType.GAM)
    # Hyperparameters for regressions
    params_gam_scale = scoring_params.get("gam_scale", False)

    models = [None for _ in range(len(X))]
    scores = [0 for _ in range(len(X))]

    for ci in range(len(X)):
        if params_score_type in GPType._member_map_.values():
            raise DeprecationWarning()

        elif params_score_type == ScoreType.GAM:

            Xtr, ytr = (data_scale(X[ci]), data_scale(y[ci].reshape(-1, 1))) if params_gam_scale else (X[ci], y[ci])
            models[ci], scores[ci] = fit_score_gam(Xtr, ytr)

        elif params_score_type == ScoreType.SPLINE:
            raise DeprecationWarning()
        else:
            raise ValueError(f"Invalid score {params_score_type}")
    return sum(scores)


def fit_score_gam(Xtr, ytr):
    gam = GAM()
    gam.fit(Xtr, ytr)
    n_splines, order = 20, 3
    mse = np.mean((gam.predict(Xtr) - ytr) ** 2)
    n = Xtr.shape[0]
    p = Xtr.shape[1] * n_splines * order
    gam.mdl_lik_train = n * np.log(mse)
    gam.mdl_model_train = 2 * p
    gam.mdl_pen_train = 0
    gam.mdl_train = gam.mdl_lik_train + gam.mdl_model_train + gam.mdl_pen_train
    return gam, gam.mdl_train


class RegressorType(Enum):
    LN = 0
    SPLN = 1
    GAM = 2
    GP = 3
    NN = 4


def fit_fun(causes, effect, reg_type, seed=42, test_indep=False):
    if causes.shape[1] == 0:
        causes = np.random.normal(size=effect.reshape(-1, 1).shape)
    if reg_type == RegressorType.LN:
        return fit_ln(causes, effect, seed, test_indep)
    elif reg_type == RegressorType.SPLN:
        return fit_spln(causes, effect, seed, test_indep)
    elif reg_type == RegressorType.GAM:
        return fit_gam(causes, effect, seed, test_indep)
    else:
        raise ValueError("Unknown regression type.")


def fit_ln(causes, effect, seed, test_indep=False):
    m = LinearRegression()
    m.fit(causes, effect)
    preds = m.predict(causes)
    resids = (effect - preds).reshape(-1, 1)
    loglik_strength = -0.5 * mean_squared_error(effect, preds) * len(effect)
    return resids, loglik_strength


def fit_spln(causes, effect, seed, test_indep=False):
    if causes.shape[1] > 1:
        raise ValueError("Polynomial spline fitting only supports single feature currently.")

    causes_flat = causes.flatten()
    knots = np.linspace(min(causes_flat), max(causes_flat), 4)  # Adjust number of knots as needed
    m = make_lsq_spline(causes_flat, effect, t=knots[1:-1])
    predictions = m(causes_flat)
    resids = (effect - predictions).reshape(-1, 1)
    loglik_strength = -0.5 * mean_squared_error(effect, predictions) * len(effect)

    return resids, loglik_strength


def fit_gam(causes, effect, seed, test_indep=False):
    m = LinearGAM().fit(causes, effect)
    preds = m.predict(causes)
    resids = (effect - preds).reshape(-1, 1)
    loglik_strength = -0.5 * mean_squared_error(effect, preds) * len(effect)

    return resids, loglik_strength
