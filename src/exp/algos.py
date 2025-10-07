import time
from abc import ABC, abstractmethod
from enum import Enum

import networkx as nx
import numpy as np
import pandas as pd

from src.baselines.sep_distances.codebase import mixed_graph as graph_lmg
from src.mixtures.greedy_equivalent_causal_mixture import ges_causal_mixture
from src.mixtures.topological_causal_mixture import TopologicalCausalMixture
from src.mixtures.util.util import compare_lmg_DAG, compare_lmg_CPDAG, nxdigraph_to_lmg, general_graph_to_lmg, \
    causaldag_to_lmg, general_graph_to_directed_edge_adj, general_graph_to_undirected_edge_adj


class DAGType(Enum):
    """ result that a method returns, DAG or (C)PDAG """
    DAG = 0
    CPDAG = 1


class OracleType(Enum):
    """Decides between different types of oracles for our method"""
    trueGtrueZ = 'trueGtrueZ'
    # Find Z
    trueGtrueK = 'trueGtrueK'
    trueGhatZ = 'trueGhatZ'
    emptyGhatZ = 'emptyGhatZ'
    fullGhatZ = 'fullGhatZ'
    # Find G
    hatGhatZ = 'hatGhatZ'
    hatGtrueZ = 'hatGtrueZ'
    SKIP = 'skip'

    def __str__(self):
        return str(self.value) if self.value != OracleType.SKIP.value else ''

    def is_G_known(self):
        return self.value in [
            OracleType.trueGtrueZ.value, OracleType.trueGtrueK.value, OracleType.trueGhatZ.value, OracleType.SKIP.value
        ]

    def is_G_empty(self): return self.value in [OracleType.emptyGhatZ.value]

    def is_G_dense(self): return self.value in [OracleType.fullGhatZ.value]

    def is_Z_known(self): return self.value in [OracleType.trueGtrueZ.value, OracleType.hatGtrueZ.value]

    def is_K_known(self): return self.value in [OracleType.trueGtrueK.value]

    def haveto_discover_G(self):
        return self.value.startswith('hatG')


class CD(Enum):
    """ causal discovery methods. """
    SKIP = 'skip'
    #
    CausalMixtures = 'causal-mixtures'
    CausalMixturesGES = 'causal-mixtures-ges'
    MixtureUTIGSP = 'mix-utigsp'
    PC_PC = 'pc-partial-correl'
    PC_KCI = 'pc-kci-partial-correl'
    FCI_PC = 'fci'
    FCI_KCI = 'fci-kci'
    GES = 'ges'
    CAM = 'cam'
    LINGAM = 'lingam'
    TOPIC = 'topic'
    SCORE = 'score'
    DAS = 'das'
    NOGAM = 'nogam'
    CAM_UV = 'cam-uv'
    R2SORT = 'r2sort'
    RANDSORT = 'randsort'
    VARSORT = 'varsort'

    def __str__(self):
        return str(self.value) if self.value != CD.SKIP.value else ''

    def get_method(self):
        """ all implemented methods"""
        if self.value == CD.CausalMixtures.value:
            return CausalMixtureMethod(self)
        elif self.value == CD.CausalMixturesGES.value:
            return CausalMixtureMethodGES(self)
        elif self.value == CD.MixtureUTIGSP.value:
            return MixtureUTIGSPMethod(self)
        elif self.value in [CD.PC_PC.value, CD.PC_KCI.value]:
            return PCMethod(self)
        if self.value in [CD.FCI_PC.value, CD.FCI_KCI.value]:
            return FCIMethod(self)
        elif self.value in [CD.GES.value]:
            return GESMethod(self)
        elif self.value in [CD.R2SORT.value, CD.RANDSORT.value, CD.VARSORT.value]:
            return SortingMethod(self)
        # elif self.value == CD.RESIT.value:
        #    return RESITMethod(self)
        elif self.value == CD.CAM_UV.value:
            return CAMUVMethod(self)
        elif self.value == CD.TOPIC.value:
            return TOPICMethod(self)
        elif self.value == CD.LINGAM.value:
            return DirectLINGAMMethod(self)
        elif self.value in [
            CD.SCORE.value,
            CD.CAM.value,
            CD.NOGAM.value,
            CD.DAS.value,
        ]:
            return TopologicalMethod(self)
        elif self.value == CD.SKIP.value:
            raise ValueError("placeholder when causal discovery is skipped")
        raise ValueError("not supported yet")

    def discovers_mixture_assignments(self):
        return self.value in [CD.CausalMixtures.value, CD.CausalMixturesGES.value, CD.MixtureUTIGSP.value]


class CausalDiscoveryMthd(ABC):

    def __init__(self, ty: CD):
        self.ty = ty
        self.metrics: dict = {}
        self.dag: nx.DiGraph = nx.DiGraph()
        self.lmg: graph_lmg.LabelledMixedGraph = graph_lmg.LabelledMixedGraph()
        self.model = None
        self.e_n_Z = {}
        self.e_Z_n = {}

    @staticmethod
    @abstractmethod
    def dag_ty() -> DAGType:
        pass

    def nm(self) -> str:
        return self.ty.value

    @abstractmethod
    def fit(self, data: np.ndarray, **kwargs):
        """ causal discovery of PAG or DAG
        :param data: data
        :param kwargs: method parameters
        :return:
        """
        pass

    def get_directed_graph(self):
        assert self.dag_ty() == DAGType.DAG
        return nx.to_numpy_array(self.dag)

    def get_labelled_mixed_graph(self):
        return self.lmg

    def get_mixture_assignment_node(self):
        assert self.ty.discovers_mixture_assignments()
        return self.e_n_Z

    def get_mixed_node_sets(self):
        assert self.ty.discovers_mixture_assignments()
        return self.e_Z_n

    def get_graph_metrics(self, true_nxg):
        true_lmg = nxdigraph_to_lmg(true_nxg)
        est_lmg = self.get_labelled_mixed_graph()

        if self.dag_ty() == DAGType.DAG:
            return compare_lmg_DAG(true_lmg, est_lmg)
        elif self.dag_ty() == DAGType.CPDAG:
            return compare_lmg_CPDAG(true_lmg, est_lmg)
        else:
            raise ValueError(self.dag_ty())


# %% Our Method ############
class CausalMixtureMethod(CausalDiscoveryMthd, ABC):
    allowed_params = ['truths', 'hybrid', 'pruning_G', 'oracle_G', 'oracle_K', 'oracle_Z', 'lg', 'k_max', 'vb']
    e_Z = {}
    Z_pairs = {}
    pprobas = {}
    idls = {}

    @staticmethod
    def dag_ty(): return DAGType.DAG

    def fit(self, X, **kwargs):
        params = {ky: val for ky, val in kwargs.items() if ky in self.allowed_params}
        params["hybrid"] = True
        top = TopologicalCausalMixture(**params)
        time_st = time.perf_counter()
        top.fit_graph_and_mixtures(X)
        self.metrics = {'time': time.perf_counter() - time_st}
        self.dag = top.topic_graph
        self.lmg = nxdigraph_to_lmg(self.dag)
        self.model = top

        # information on reconstructed mixing variables, targeted observed variable sets
        self.e_n_Z = top.e_n_Z
        self.e_Z_n = top.e_Z_n
        self.e_Z = top.e_Z
        self.Z_pairs = top.Z_pairs
        self.pprobas = top.pprobas
        self.idls = top.idls

class CausalMixtureMethodGES(CausalDiscoveryMthd, ABC):
    allowed_params = ['oracle_K', 'oracle_Z', 'k_max', 'vb']
    e_Z = {}
    Z_pairs = {}
    pprobas = {}
    idls = {}

    @staticmethod
    def dag_ty(): return DAGType.CPDAG

    def fit(self, X, **kwargs):
        from causallearn.graph import GeneralGraph
        params = {ky: val for ky, val in kwargs.items() if ky in self.allowed_params}
        k_max = params.get('k_max', 5)
        ges_score = "local_score_latent_BIC"

        time_st = time.perf_counter()
        ges_obj = ges_causal_mixture(X, ges_score, parameters=params)
        # -------
        # ges_obj['G']: learned causal graph, where ges_obj['G'].graph[j,i]=1 and ges_obj['G'].graph[i,j]=-1 indicates  i --> j ,
        #            ges_obj['G'].graph[i,j] = ges_obj['G'].graph[j,i] = -1 indicates i --- j.

        gg: GeneralGraph = ges_obj['G']
        self.lmg = general_graph_to_lmg(gg)
        self.model = ges_obj

        # reconstruct the mixing variables under G
        adj = general_graph_to_undirected_edge_adj(gg)
        hypparams = dict(oracle_Z=False, oracle_K=False, oracle_G=False, k_max=params.get('k_max', 5))
        top = TopologicalCausalMixture(**hypparams)
        top.fit_Z_given_G(X, adj, skip_pruning=True) #pruning is only for the power-speci experiments
        self.metrics = {'time': time.perf_counter() - time_st}

        self.model = {'ges-with-latent-bic': ges_obj, 'mixture-variable-extraction': top}

        # information on reconstructed mixing variables, targeted observed variable sets
        self.e_n_Z = top.e_n_Z
        self.e_Z_n = top.e_Z_n
        self.e_Z = top.e_Z
        self.Z_pairs = top.Z_pairs
        self.pprobas = top.pprobas
        self.idls = top.idls

# %% BASELINES ############
class MixtureUTIGSPMethod(CausalDiscoveryMthd, ABC):
    @staticmethod
    def dag_ty(): return DAGType.DAG

    def fit(self, X, **kwargs):
        if "intv_args_dict" not in kwargs or "args" not in kwargs: raise Warning(
            "Usage (MixtureUTIGSP): provide hyperparameters in 'args'")
        model = None
        time_st = time.perf_counter()

        # Step 1: Disentanglement/mixture modelling
        from src.baselines.mixture_mec.mixture_solver import GaussianMixtureSolver
        mixture_samples = X
        intv_args_dict = kwargs.get("intv_args_dict", {})
        args = kwargs.get("args", {})

        gSolver = GaussianMixtureSolver(args["dtype"])
        err, intv_args_dict, weight_precision_error, est_num_comp, gm_score_dict, gm \
            = gSolver.mixture_disentangler(
            args["num_tgt_prior"],
            intv_args_dict,
            mixture_samples,
            args["gmm_tol"],
            args["cutoff_drop_ratio"],
        )

        # Step 2: structure learning and intervention target identification
        est_dag, intv_args_dict, oracle_est_dag, igsp_est_dag, intv_base_est_dag \
            = gSolver.identify_intervention_utigsp(
            intv_args_dict, args["stage2_samples"])
        self.metrics = {'time': time.perf_counter() - time_st}

        # Result extraction
        est_tgts = [
            node_i for node_i in range(mixture_samples.shape[1]) if
            any(["est_tgt" in intv_args_dict[ky] and node_i in intv_args_dict[ky]["est_tgt"] and ky != "obs" for ky in
                 intv_args_dict])]
        self.lmg = causaldag_to_lmg(est_dag)
        self.model = model
        self.e_Z_n = [gm.predict(mixture_samples) if node_i in est_tgts else np.zeros(mixture_samples.shape[0]) for
                      node_i in range(mixture_samples.shape[1])]
        self.e_n_Z = [est_tgts]


class TOPICMethod(CausalDiscoveryMthd, ABC):
    @staticmethod
    def dag_ty(): return DAGType.DAG

    def fit(self, X, **kwargs):
        from src.mixtures.topological_causal_mixture import TopologicalCausalMixture
        kwargs["hybrid"] = False  # makes sure this does not use the latent-aware BIC, but a latent-unaware score
        top = TopologicalCausalMixture(**kwargs)
        time_st = time.perf_counter()
        self.metrics = {'time': time.perf_counter() - time_st}
        self.dag = top.fit_graph(X)
        self.lmg = nxdigraph_to_lmg(self.dag)
        self.model = top


class TopologicalMethod(CausalDiscoveryMthd, ABC):
    @staticmethod
    def dag_ty(): return DAGType.DAG

    def fit(self, X, **kwargs):
        from src.baselines.dodiscover import make_context

        from src.baselines.dodiscover.toporder import SCORE, CAM, NoGAM, DAS
        model = SCORE() if self.ty.value == CD.SCORE.value else \
            CAM() if self.ty.value == CD.CAM.value else \
                NoGAM() if self.ty.value == CD.NOGAM.value else \
                    DAS() if self.ty.value == CD.DAS.value \
                        else None
        score_context = make_context().variables(data=pd.DataFrame(X)).build()
        time_st = time.perf_counter()
        model.learn_graph(pd.DataFrame(X), score_context)
        self.metrics = {'time': time.perf_counter() - time_st}

        self.dag = model.graph_
        self.lmg = nxdigraph_to_lmg(self.dag)
        self.model = model


class PCMethod(CausalDiscoveryMthd, ABC):
    """causal-learn implementation"""

    @staticmethod
    def dag_ty(): return DAGType.CPDAG

    def fit(self, X, **kwargs):
        from causallearn.search.ConstraintBased.PC import pc
        indep_test = 'mv_fisherz' if self.ty == CD.PC_PC else 'kci'
        time_st = time.perf_counter()
        pc_obj = pc(X, indep_test=indep_test)
        # -------
        # G : a CausalGraph object, where G.graph[j,i]=1 and G.graph[i,j]=-1 indicates  i --> j ,
        #                G.graph[i,j] = G.graph[j,i] = -1 indicates i --- j,
        #                G.graph[i,j] = G.graph[j,i] = 1 indicates i <-> j.
        gg = pc_obj.G
        self.lmg = general_graph_to_lmg(gg)
        self.metrics = {'time': time.perf_counter() - time_st}
        self.model = pc_obj


class GESMethod(CausalDiscoveryMthd, ABC):
    @staticmethod
    def dag_ty(): return DAGType.CPDAG

    def fit(self, X, **kwargs):
        # cdt implementation:
        # import cdt.causality.graph as algs
        # obj = algs.GES()
        # datafr = pd.DataFrame(data)
        # self.untimed_graph = obj.predict(datafr)

        from causallearn.search.ScoreBased.GES import ges
        from causallearn.graph import GeneralGraph
        ges_score = "local_score_BIC" if self.ty == CD.GES else None
            #"local_score_CV_multi" if self.ty == CD.GGES_CV else \
            #    "local_score_marginal_multi" if self.ty == CD.GGES_MARG else None

        time_st = time.perf_counter()
        ges_obj = ges(X, ges_score)
        # -------
        # ges_obj['G']: learned causal graph, where ges_obj['G'].graph[j,i]=1 and ges_obj['G'].graph[i,j]=-1 indicates  i --> j ,
        #            ges_obj['G'].graph[i,j] = ges_obj['G'].graph[j,i] = -1 indicates i --- j.

        gg: GeneralGraph = ges_obj['G']
        self.lmg = general_graph_to_lmg(gg)
        self.metrics = {'time': time.perf_counter() - time_st}
        self.model = ges_obj


class SortingMethod(CausalDiscoveryMthd, ABC):
    @staticmethod
    def dag_ty(): return DAGType.DAG

    def fit(self, X, **kwargs):
        from src.baselines.CausalDisco.baselines import (
            r2_sort_regress, var_sort_regress, random_sort_regress
        )
        fun = r2_sort_regress if self.ty == CD.R2SORT else \
            var_sort_regress if self.ty == CD.VARSORT else \
                random_sort_regress if self.ty == CD.RANDSORT else None
        time_st = time.perf_counter()
        self.dag = nx.from_numpy_array(fun(X), create_using=nx.DiGraph)
        self.lmg = nxdigraph_to_lmg(self.dag)
        self.metrics = {'time': time.perf_counter() - time_st}


class FCIMethod(CausalDiscoveryMthd, ABC):
    @staticmethod
    def dag_ty(): return DAGType.CPDAG

    def fit(self, X, **kwargs):
        from causallearn.search.ConstraintBased.FCI import fci
        indep_test = 'fisherz' if self.ty == CD.FCI_PC else 'kci'
        time_st = time.perf_counter()
        graph, edges = fci(X, independence_test_method=indep_test)
        # from causallearn.search.ConstraintBased.FCI:
        #     graph : a GeneralGraph object, where graph.graph[j,i]=1 and graph.graph[i,j]=-1 indicates  i --> j ,
        #                     graph.graph[i,j] = graph.graph[j,i] = -1 indicates i --- j,
        #                     graph.graph[i,j] = graph.graph[j,i] = 1 indicates i <-> j,
        #                     graph.graph[j,i]=1 and graph.graph[i,j]=2 indicates  i o-> j.
        # currently don't evaluate the following edges that could point to latent mixing variables: graph.graph[j,i]=1 and graph.graph[i,j]=2 indicates  i o-> j.
        gg = graph
        self.lmg = general_graph_to_lmg(gg)
        self.metrics = {'time': time.perf_counter() - time_st}
        self.model = graph, edges


class CAMUVMethod(CausalDiscoveryMthd, ABC):
    """ causal discovery toolbox implementation """

    @staticmethod
    def dag_ty():
        return DAGType.DAG

    def fit(self, X, **kwargs):
        num_explanatory_vals = kwargs.get("num_explanatory_vals", 3)
        alpha = kwargs.get("alpha", 0.05)
        print("CAM-UV: Setting num_explanatory_vals to: ", num_explanatory_vals, ", alpha: ", alpha)

        from causallearn.search.FCMBased.lingam import CAMUV

        time_st = time.perf_counter()

        # Usage
        # P: P[i] contains the indices of the parents of Xi
        # U: The indices of variable pairs having UCPs or UBPs

        P, U = CAMUV.execute(X, alpha, num_explanatory_vals)
        self.metrics = {'time': time.perf_counter() - time_st}

        dag = nx.DiGraph()
        dag.add_nodes_from(set(range(len(P))))
        for i, result in enumerate(P):
            if not len(result) == 0:
                print("child: " + str(i) + ",  parents: " + str(result))
                for j in result:
                    dag.add_edge(j, i)
        print("CAM-UV: evaluate indices U")
        self.dag = dag
        self.lmg = nxdigraph_to_lmg(self.dag)
        self.model = P, U


class GLOBEMethod(CausalDiscoveryMthd, ABC):
    @staticmethod
    def dag_ty(): return DAGType.DAG

    def fit(self, X, **kwargs):
        from src.baselines.globe import GlobeWrapper

        max_interactions = kwargs.get("max_interactions", 3)
        print("Setting max interactions to: ", max_interactions)

        model = GlobeWrapper(max_interactions, False, True)
        data = pd.DataFrame(X)
        data.to_csv("temp.csv", header=False, index=False)
        model.loadData("temp.csv")
        time_st = time.perf_counter()
        adjacency = model.run()
        self.metrics = {'time': time.perf_counter() - time_st}

        self.dag = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
        self.lmg = nxdigraph_to_lmg(self.dag)
        self.model = model


class ICALINGAMMethod(CausalDiscoveryMthd, ABC):
    """causallearn implementation"""

    @staticmethod
    def dag_ty(): return DAGType.DAG

    def fit(self, X, **kwargs):
        from causallearn.search.FCMBased import lingam

        model = lingam.ICALiNGAM()
        time_st = time.perf_counter()
        model.fit(X)
        self.metrics = {'time': time.perf_counter() - time_st}

        self.dag = nx.from_numpy_array(model.adjacency_matrix_, create_using=nx.DiGraph)
        self.lmg = nxdigraph_to_lmg(self.dag)
        self.model = model


class DirectLINGAMMethod(CausalDiscoveryMthd, ABC):
    """causallearn implementation"""

    @staticmethod
    def dag_ty(): return DAGType.DAG

    def fit(self, X, **kwargs):
        from causallearn.search.FCMBased import lingam

        model = lingam.DirectLiNGAM()  # random_state, prior_knowledge, apply_prior_knowledge_softly, measure)
        time_st = time.perf_counter()
        model.fit(X)
        self.metrics = {'time': time.perf_counter() - time_st}

        self.dag = nx.from_numpy_array(model.adjacency_matrix_, create_using=nx.DiGraph)
        self.lmg = nxdigraph_to_lmg(self.dag)
        self.model = model


class LINGAMMethod(CausalDiscoveryMthd, ABC):
    """cdt implementation"""

    @staticmethod
    def dag_ty(): return DAGType.DAG

    def fit(self, X, **kwargs):
        from cdt.causality.graph import LiNGAM
        model = LiNGAM()
        time_st = time.perf_counter()
        dag = model.predict(pd.DataFrame(X))
        self.metrics = {'time': time.perf_counter() - time_st}
        self.dag = dag
        self.lmg = nxdigraph_to_lmg(self.dag)
        self.model = model
