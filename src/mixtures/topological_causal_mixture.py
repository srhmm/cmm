import itertools

import networkx as nx
import numpy as np
import sklearn

from src.mixtures.memoized_edge_score import MemoizedEdgeScore
from src.mixtures.mixing.mixing import MixingType
from src.mixtures.util.util import is_insignificant, compare_Z, nxdigraph_to_lmg, compare_lmg_DAG
from src.mixtures.util.utils_idl import exp_mutual_info_score, pi_xor_pessimistic, \
    expected_sampled_mutual_info_score, expected_sampled_adjusted_mutual_info_score, expected_smi, \
    soft_mutual_information, _get_true_idl_Z
from src.mixtures.util.visu import visu_pair_mi, visu_node_pproba, visu_node_idls, \
    visu_node_idl, visu_node_true_idl, visu_pproba_dens


class TopologicalCausalMixture:
    mixing_type: MixingType
    topic_graph: nx.DiGraph
    hybrid: bool
    oracle_Z: bool
    oracle_K: bool
    oracle_G: bool
    oracle_order: bool
    truths: dict
    topological_order: list
    extra_refinement: True
    true_graph: None
    true_top_order: []
    candidates: []
    k_max: int
    scores: MemoizedEdgeScore
    lg: None
    vb: int

    def __init__(self, **kwargs):
        r""" Topological Causal Discovery with mixtures of causal mechanisms
        :param optargs: optional arguments

        :Keyword Arguments:
        * *mixing_type* (``MixingType``) -- type of mixture model inference (EM algo)
        * *truths* (``nx.DiGraph``) -- for oracle versions, w entries 't_A', 't_Z', 't_n_Z'
        * *oracle_G* (``bool``) -- known graph
        * *oracle_order* (``bool``) -- known top order
        * *oracle_K* (``bool``) -- known n mixtures per confounder
        * *oracle_Z* (``bool``) -- known assignments per node (debug)
        * *hybrid* (``bool``) -- fit mixtures during graph search
        * *k_max* (``int``) -- max n mixtures
        * *mi_thresh* (``float``) -- sigificance thresh MI pairs of nodes
        * *smi_thresh* (``float``) -- sigificance thresh soft MI pairs of nodes
        * *bic_thresh* (``float``) -- sigificance thresh edge score
        * *lambda_mix* (``float``) -- regularization param for mixing penalty in score
        * *use_smi* (``float``) -- use soft MI instead of MI to detect confounding between node pair
        * *lg* (``logging``) -- logger if verbosity>0
        * *vb* (``int``) -- verbosity level
        """
        self.defaultargs = {
            "mixing_type": MixingType.MIX_LIN,
            "truths": dict(),
            "oracle_G": False,
            "oracle_order": False,
            "oracle_K": False,
            "oracle_Z": False,
            "k_max": 5,
            "hybrid": True,
            "mi_thresh": 0.03,
            "smi_thresh": 0.01,
            "bic_thresh": 0,
            "lambda_mix": 1,
            "use_smi": False,
            "extra_refinement": True,
            "lg": None,
            "vb": 0}

        self.__dict__.update((k, v) for k, v in self.defaultargs.items() if k not in kwargs.keys())
        self.__dict__.update((k, v) for k, v in kwargs.items())  # if k in self.defaultargs.keys())

        def _info(st, strength=0):
            (self.lg.info(st) if self.lg is not None else print(st)) if self.vb + strength > 0 else None

        self._info = _info
        self.is_true_edge = (lambda i: lambda j: "") if 'true_g' not in self.truths else \
            (lambda node: lambda other: 'causal' if self.truths['true_g'].has_edge(node, other) else (
                'rev' if self.truths['true_g'].has_edge(other, node) else 'spurious'))
        assert not self.oracle_order or 'true_g' in self.truths
        self.true_top_order = [] if 'true_g' not in self.truths or not self.oracle_order else list(
            nx.topological_sort(self.truths['true_g']))

        if self.oracle_Z: self.use_smi = False  # no pprobas

        self.topic_graph = nx.DiGraph()
        self.topological_order = []
        self._add_greedily = False

        self.idls = {}
        self.pprobas = {}
        self.div_measures = {}

        self.fitted_graph, self.fitted_mixing = False, False

    def fit_graph_and_mixtures(self, X):
        self._info(f"\t*** CausalMixtures ({self.mixing_type.value}, mix. during ordering: {self.hybrid}) ***")

        self.fit_graph(X)
        self._fit_Z_nodes(X)
        self._fit_Z_sets()

        self.fitted_graph, self.fitted_mixing = True, True
        return self.topic_graph, self.idls, self.pprobas, self.e_n_Z, self.e_Z

    def fit_graph(self, X):
        self._check_X(X)
        self.scores = MemoizedEdgeScore(self.mixing_type, self.X, **self.get_scoring_params())

        if self.oracle_G:
            assert 'true_g' in self.truths, 'need truths if oracle_G'
            self.topic_graph = self.truths['true_g']
            return self.topic_graph

        self.topic_graph.add_nodes_from(range(self.N))
        self.candidates = list(range(self.N))
        self.order_nodes()
        self.fitted_graph = True  # before _info_graph because it shows metrics
        self._info_graph()

        return self.topic_graph

    def fit_Z_given_G(self, X, graph_adj, skip_pruning=False, skip_sets=False):
        graph = nx.from_numpy_array(graph_adj, create_using=nx.Graph)
        self._info(f"\t*** Causal Mixture Modeling (given DIgraph w {len(graph.edges())} edges) ***")
        self._check_X(X)
        self.topic_graph = graph.copy()
        self.scores = MemoizedEdgeScore(self.mixing_type, self.X, **self.get_scoring_params())
        self.fitted_mixing = True

        self._fit_Z_nodes(X)
        if not skip_sets: self._fit_Z_sets()
        if skip_pruning: return

        scoring_params = self.get_scoring_params()
        scoring_params["hybrid"] = True  # enforce that we score each domain separately here
        self.scores = MemoizedEdgeScore(self.mixing_type, self.X, **scoring_params)
        self.prune_spurious_edges(self.topic_graph.copy())
        self._fit_Z_nodes(X)
        if not skip_sets: self._fit_Z_sets()

    def _fit_Z_nodes(self, X):
        for nodei in self.topic_graph.nodes:
            cov = [i for i in self.topic_graph.predecessors(nodei)] if nx.is_directed_acyclic_graph(self.topic_graph)  else [i for i in self.topic_graph.neighbors(nodei)]
            self.idls[nodei], self.pprobas[nodei], self.div_measures[nodei] = self.scores.idl_edge(nodei, cov)
            if all([ky in self.truths for ky in ['t_A', 't_Z', 't_n_Z']]):
                true_idl = _get_true_idl_Z(np.where(self.truths['t_A'][:, nodei] != 0)[0], nodei, self.truths['t_A'],
                                           self.truths['t_Z'], self.truths['t_n_Z'], X.shape[0])
                ami = sklearn.metrics.adjusted_mutual_info_score(true_idl, self.idls[nodei])
                self._info(
                    f"\t\tNode {nodei} | {cov}: {ami:.2f} k^={len(np.unique(self.idls[nodei]))} k*={len(np.unique(true_idl))}",
                    -1)  # -1
                if self.vb >= 2: self.visu_scatter_mixing_assignment_node(nodei)

        self.e_Z_n = self.idls

    def _fit_Z_sets(self, n_nodes=None):
        if n_nodes is None: n_nodes = len(self.topic_graph.nodes)
        # adj_A = nx.to_numpy_array(self.topic_graph)
        self.Z_pairs = []
        self.Z_pairs_scores = {}
        self.confd_A = np.zeros((n_nodes, n_nodes))
        alt_A = np.zeros((n_nodes, n_nodes))

        # Jointly mixed nodes (confounding)
        for (i, j) in itertools.combinations(set(range(n_nodes)), 2):
            Za, Zb = self.idls[i], self.idls[j]
            pa, pb = self.pprobas[i], self.pprobas[j]

            mi = sklearn.metrics.mutual_info_score(Za, Zb)
            ami = sklearn.metrics.adjusted_mutual_info_score(Za, Zb)
            smi = soft_mutual_information(pa, pb) if not self.oracle_Z else mi
            # asmi = soft_adjusted_mutual_information(pa, pb)
            emi, _ = exp_mutual_info_score(Za, Zb)
            emi_samp = expected_sampled_mutual_info_score(Za, Zb)
            eami = expected_sampled_adjusted_mutual_info_score(Za, Zb)
            esmi, _ = expected_smi(pa, pb, len(Za)) if not self.oracle_Z else (emi, 0)

            mi_val = 0 if len(np.unique(Za)) == 1 or len(np.unique(Zb)) == 1 else \
                mi if mi > emi and mi > self.mi_thresh else 0
            smi_val = 0 if len(np.unique(Za)) == 1 or len(np.unique(Zb)) == 1 else \
                smi if smi > esmi and smi > self.smi_thresh else 0

            self.confd_A[i][j] = smi_val if self.use_smi else mi_val
            alt_A[i][j] = mi_val if self.use_smi else smi_val
            is_confounded = self.confd_A[i][j] > 0
            if is_confounded: self.Z_pairs.append((i, j))
            self.Z_pairs_scores[f"{i}-{j}"] = {"ami": ami, "mi": mi, "smi": smi, "emi": emi, "esmi": esmi}

        # Dependently mixed nodes (confounding)
        componentG = nx.from_numpy_array(self.confd_A, create_using=nx.Graph)
        confd_targets = [set(n_set) for n_set in nx.connected_components(componentG) if len(n_set) > 1]

        self.confd_targets_alt = [set(n_set) for n_set in
                                  nx.connected_components(nx.from_numpy_array(alt_A, create_using=nx.Graph)) if
                                  len(n_set) > 1]

        # Independently mixed nodes (independent change)
        for i in list(range(n_nodes)):
            if not any([i in n_set for n_set in confd_targets]) and len(np.unique(self.idls[i])) > 1:
                confd_targets.append({i})

            if not any([i in n_set for n_set in self.confd_targets_alt]) and len(np.unique(self.idls[i])) > 1:
                self.confd_targets_alt.append({i})

        # Simple aggregation of confounder labels per node set
        confd_idls = []
        for n_set in confd_targets:
            confd_avg = np.zeros(self.X.shape[0])
            for node_i in n_set:
                confd_idl = self.idls[node_i]  # todoself.get_edge_confd_A(node_i, adj_A)
                confd_avg = pi_xor_pessimistic(confd_idl, confd_avg)
            confd_idls.append(confd_avg)

        # per node set
        self.e_Z = confd_idls
        self.e_n_Z = confd_targets
        assert len(self.e_n_Z) == len(self.e_n_Z)

        # per node
        self.e_Z_n = self.idls
        self.e_Zp_n = self.pprobas
        assert len(self.e_Z_n) == len(self.e_Zp_n)  # == len(self.topic_graph.nodes)
        self._info(f"\t** Confd node sets: {self.e_n_Z}")

    # %% GRAPH PRUNING
    def local_pruning(self, node, parents):
        assert self.scores is not None and self.scores.hybrid  # make sure Zs will be used during scoring
        parent_subset = []
        for parent in parents:
            # assert (parent, node) in self.topic_graph.edges, f"edge {parent}->{node} not in graph"
            if (parent, node) in self.topic_graph.edges:
                self.topic_graph.remove_edge(parent, node)
        for parent in parents:  # try score-pruning under the node's true partition
            gain = self._addition_gain(node, parent)  # during scoring&fitting: values of Z will be used
            if self._significant(gain) or self._add_greedily:
                self._add_edge(parent, node, gain=float(gain))
                parent_subset.append(parent)
        return parent_subset

    def prune_spurious_edges(self, graph):
        self.scores.hybrid = True  # enforce that we score each domain separately here
        self._info(f"\t** Edge Pruning ")
        for nodei, node in enumerate(graph.nodes()):
            pre = graph.predecessors(node) if nx.is_directed_acyclic_graph(graph) else graph.neighbors(node)
            parents_spurious = [nodem for nodem in pre]
            idl = self.e_Z_n[nodei]
            parent_subset = self.local_pruning(nodei, parents_spurious)

            for nodej in parents_spurious:
                pair_confounded = ((nodej, nodei) in self.Z_pairs or (nodei, nodej) in self.Z_pairs)
                prune = nodej not in parent_subset
                if 'true_g' in self.truths:
                    correct = "(keeping true edge correctly)" if not prune and (nodej, nodei) in self.truths[
                        'true_g'].edges else \
                        "(pruning ncaus. edge correctly)" if prune and (nodej, nodei) not in self.truths[
                            'true_g'].edges else \
                            "(pruning true edge erroneously)" if prune and (nodej, nodei) in self.truths[
                                'true_g'].edges else "(keeping ncaus. edge erroneously)"
                else:
                    correct = ""
                self._info(
                    f"\t\tPair {nodej}->{nodei}: {'cfd' if pair_confounded else '1. uncfd'}  {'PRUNED' if prune else 'keep'} {correct}",
                    -2)

    # %% GRAPH SEARCH: TOPOLOGICAL ORDERING
    def order_nodes(self):
        it = 0
        while it < self.N:
            source = self.get_next_node(self.candidates if not self.oracle_order else self.true_top_order[it])
            self.candidates.remove(source)
            self.topological_order.append(source)
            it += 1
            self._info(f"\t{it}. Source: {source}\t current {self.topological_order}, true {self.true_top_order}", -2)

            self.add_edges(source)
            self.refine_edges(source)

        if self.extra_refinement:
            self.refinement_phase()

    def add_edges(self, source):
        for node in self.candidates:
            if node in self.topological_order or node == source or self.has_cycle(source, node):
                continue
            gain = self._addition_gain(node, source)
            if self._significant(gain) or self._add_greedily:
                self._add_edge(source, node, gain=float(gain))

    def _addition_gain(self, node, source):
        parents = list(self.topic_graph.predecessors(node)).copy() if nx.is_directed_acyclic_graph(self.topic_graph) else  list(self.topic_graph.neighbors(node)).copy()
        old_score = self._score(parents, node)
        parents.append(source)
        new_score = self._score(parents, node)
        gain = self._gain(new_score, old_score)
        return gain

    def refine_edges(self, source):
        parents = list(self.topic_graph.predecessors(source))
        n_removed = 0
        while n_removed < len(parents):
            removed_found, removed_parent = self.refine_step(source, parents)

            if removed_parent is not None:
                self._remove_edge(removed_parent, source)
                parents.remove(removed_parent)
                n_removed += 1
            else:
                break

    def refine_step(self, source, parents):
        removed_found, best_parent, best_diff = False, None, np.inf
        old_score = self._score(parents, source)

        for parent in parents:
            new_parents = parents.copy()
            new_parents.remove(parent)
            if len(new_parents) == 0:
                continue
            new_score = self._score(new_parents, source)
            diff = old_score - new_score  # new_score - old_score

            if diff < best_diff and self._significant(diff):
                best_diff = diff
                best_parent = parent
                removed_found = True
        return removed_found, best_parent

    # %% GRAPH SEARCH: EDGE-GREEDY
    # todo future

    # %% GRAPH AND SCORING UTILS
    def _add_edge(self, parent, child, vb=-2, gain=None):
        self.topic_graph.add_edge(parent, child)
        self._info(
            f"\tAdding {self.is_true_edge(parent)(child)} edge {parent} -> {child} {'' if gain is None else f': gain {gain:.2f}'}",
            vb)

    def _info_graph(self, vb=-1, gain=None):
        if 'true_g' in self.truths:
            self._info(
                f"\tResult: {', '.join([f'{ky}:{val:.2f}' for ky, val in self.get_metrics_graph(self.truths['true_g']).items()])}\nEdges:")
        for (parent, child) in self.topic_graph.edges:
            self._info(f"\t\t{self.is_true_edge(parent)(child)} edge {parent} -> {child}", vb)

    def _remove_edge(self, parent, child, vb=-2):
        self.topic_graph.remove_edge(parent, child)
        self._info(f"\tRemoving {self.is_true_edge(parent)(child)} edge {parent} -> {child}", vb)

    def _score(self, parents, child, vb=-3):
        if self.hybrid:
            _, _, idl_dict = self.scores.idl_edge(child, parents)
        else:
            idl_dict = dict()
        score = self.scores.score_edge(child, parents, idl_dict)
        if len(parents):
            self._info(
                f"\tScoring {'&'.join([self.is_true_edge(parent)(child) for parent in parents])} edge {parents} -> {child}\t{score}={idl_dict.get('bic', score)}+{idl_dict.get('entropy', 0)}",
                vb)
        return score

    def _gain(self, new_score, old_score):
        return old_score - new_score

    def _improvement(self, new_score, old_score):
        return new_score - old_score

    def _significant(self, gain):
        return gain > self.bic_thresh

    def _check_X(self, X):
        self.X = X
        self.D, self.N = X.shape
        assert self.D > 0 and 0 < self.N < self.D

    def has_cycle(self, source, node):
        G_hat = self.topic_graph.copy()
        G_hat.add_edge(source, node)
        try:
            _ = nx.find_cycle(G_hat, orientation="original")
        except nx.exception.NetworkXNoCycle:
            return False
        return True

    def get_next_node(self, candidates):
        if self.oracle_order:
            n = len(self.topological_order)
            self._info(f"\tTrue Next Node: {self.true_top_order[n]}", -2)
            return self.true_top_order[n]

        improvement = self.get_improvement_matrix(self.topic_graph, candidates)
        delta = improvement - improvement.T
        # find the node with the smallest possible delta
        np.fill_diagonal(delta, -np.inf)
        best_delta = np.max(delta, axis=1)
        worst = np.argmin(best_delta)

        self._info(f"\tNext Node: {candidates[worst]}, order {self.topological_order} ", -2)
        return candidates[worst]

    def get_improvement_matrix(self, graph, candidates):
        improvement_matrix = np.zeros((len(candidates), len(candidates)))
        for cause in candidates:
            for effect in candidates:
                if cause == effect:
                    continue
                parents = list(graph.predecessors(effect))
                old_score = self._score(
                    parents, effect)
                parents.append(cause)
                new_score = self._score(parents, effect)
                improvement_matrix[candidates.index(cause), candidates.index(effect)] = \
                    self._improvement(new_score, old_score)
        return improvement_matrix

    def refinement_phase(self, min_parent_set_size=0):
        # smallest subset of parents with insignificant score gain
        for j in self.topic_graph.nodes:
            parents = list(self.topic_graph.predecessors(j))
            if len(parents) == 0:
                continue

            best_size = np.inf
            arg_max = None

            old_score = self._score(parents, j)
            old_parents = parents.copy()

            for k in range(min_parent_set_size, len(parents) + 1 - 1):
                parent_sets = itertools.combinations(parents, k)
                for parent_set in parent_sets:

                    new_score = self._score(parent_set, j)
                    gain = self._gain(new_score, old_score)

                    if is_insignificant(np.abs(gain)) and len(parent_set) < best_size:  # favor smaller parent sets
                        best_size = len(parent_set)
                        arg_max = parent_set

            if arg_max is None:
                continue
            self._info(f'\trefine {parents} to {arg_max} -> {j}', -2)
            for p in old_parents:
                if p not in arg_max:
                    self._remove_edge(p, j)

    # %% THRESHOLDING of mi values
    def get_Z_sets_thresholds(self, met='mi', thresh_range=np.linspace(0, 0.05, 10)):
        adj_A = nx.to_numpy_array(self.topic_graph)
        print(f"Using: {met}")
        for thresh in thresh_range:
            A = np.zeros(adj_A.shape)
            for (i, j) in itertools.combinations(self.topic_graph.nodes, 2):
                Za, Zb = self.idls[i], self.idls[j]
                pa, pb = self.pprobas[i], self.pprobas[j]

                mi = sklearn.metrics.mutual_info_score(Za, Zb)
                ami = sklearn.metrics.adjusted_mutual_info_score(Za, Zb)
                smi = soft_mutual_information(pa, pb)
                emi, _ = exp_mutual_info_score(Za, Zb)  # expected_sampled_mutual_info_score(Za, Zb)
                eami = expected_sampled_adjusted_mutual_info_score(Za, Zb)
                esmi, _ = expected_smi(pa, pb, len(Za))

                used_i, used_ei = (mi, emi) if met == 'mi' else (ami, eami) if met == 'ami' else (smi, esmi)
                A[i][j] = 0 if len(np.unique(Za)) == 1 or len(np.unique(Zb)) == 1 else \
                    used_i if used_i > used_ei and used_i > thresh else 0

            confd_targets = [set(n_set) for n_set in
                             nx.connected_components(nx.from_numpy_array(A, create_using=nx.Graph)) if len(n_set) > 1]
            for i in list(self.topic_graph.nodes):
                if not any([i in n_set for n_set in confd_targets]) and len(
                    np.unique(self.idls[i])) > 1: confd_targets.append({i})
            print(f"Threshold: {thresh}, targets: {confd_targets}")

    # %% VISU
    def visu_pproba_dens(self, nodei):
        if len(np.unique(self.idls[nodei])) > 1: visu_pproba_dens(self.pprobas[nodei])

    def visu_heatmatrix_nodepair_MI(self, **kwargs):
        visu_pair_mi(self.e_Z_n, self.pprobas, soft=False, **kwargs)
        if not self.oracle_Z: visu_pair_mi(self.e_Z_n, self.pprobas, soft=True, **kwargs)

    def visu_scatter_mixing_assignments(self, **kwargs):
        visu_node_idls(self.truths['true_g'], self.X, self.e_Z_n, **kwargs)

    def visu_scatter_true_assignments(self, nodei, **kwargs):
        assert 'true_g' in self.truths and 't_n_Z' in self.truths and 't_Z' in self.truths
        visu_node_true_idl(self.X, nodei, list(self.truths['true_g'].predecessors(nodei)), self.truths['t_n_Z'],
                           self.truths['t_Z'], method_idf='_trueGtrueZ', **kwargs)

    def visu_scatter_mixing_confidence(self, **kwargs):
        visu_node_pproba(self.truths['true_g'], self.X, self.e_Z_n, self.pprobas, **kwargs)

    def visu_scatter_mixing_assignment_node(self, nodei, **kwargs):
        visu_node_idl(self.X, nodei, list(self.truths['true_g'].predecessors(nodei)), self.idls[nodei], **kwargs)
        visu_node_true_idl(self.X, nodei, list(self.truths['true_g'].predecessors(nodei)), self.truths['t_n_Z'],
                           self.truths['t_Z'],
                           method_idf='_trueGtrueZ', **kwargs)

    def visu_scatter_mixing_fit(self, node, parents, k_range, **kwargs):
        from src.mixtures.mixing.mixing import fit_functional_mixture
        idl, _, _ = fit_functional_mixture(self.X, node, parents, k_range, None, None, lg=None)
        visu_node_idl(self.X, node, parents, idl, method_idf="_givenPAestimZ", **kwargs)

    # %% remove
    def get_scoring_params(self):
        return dict(
            hybrid=self.hybrid,
            oracle_Z=self.oracle_Z,
            oracle_K=self.oracle_K,
            t_A=self.truths.get('t_A', None),
            t_Z=self.truths.get('t_Z', None),
            t_n_Z=self.truths.get('t_n_Z', None),
            k_max=self.k_max,
            lambda_mix=self.lambda_mix,
            lg=self.lg,
        )

    # %% EVALUATION
    def get_metrics_graph(self, true_nxg: nx.DiGraph):
        """ Evaluate causal graph  """
        assert self.fitted_graph
        true_lmg = nxdigraph_to_lmg(true_nxg)
        est_lmg = nxdigraph_to_lmg(self.topic_graph)
        return compare_lmg_DAG(true_lmg, est_lmg)

    def get_metrics_mixing(self, true_A: np.ndarray, true_Z: list, true_n_Z: list):
        """ Evaluate recovery of mixing assignments, number of "confounders"/mixing nodes, and confounded/mixed node sets

        :param true_A: true adjacency
        :param true_Z: list of true confounder labels (categorical)
        :param true_n_Z: list of sets: the targeted nodes for each confounder
        :return: metrics dict
        """
        assert self.fitted_mixing
        return compare_Z(
            self.X.shape[0], true_A, nx.to_numpy_array(self.topic_graph), true_Z, true_n_Z, self.e_Z, self.Z_pairs,
            self.e_n_Z, self.e_Z_n, self.pprobas)
