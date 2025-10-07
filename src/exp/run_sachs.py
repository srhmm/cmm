import os

import networkx as nx
import numpy as np
import pandas as pd

from src.baselines.mixture_mec.mixture_solver import GaussianMixtureSolver
from src.exp.algos import CD, CausalMixtureMethodGES
from src.exp.gen.sachs.sachs_utils import generate_mixture_sachs, sachs_eval_iv_targets, write_out_metrs
from src.mixtures.util.util import causaldag_to_lmg, nxdigraph_to_lmg, compare_lmg_CPDAG, compare_lmg_DAG




def sachs_run_causal_discovery(mixture_samples, cd_mthd, k_max):
    print(f'Method: {cd_mthd.value}')
    cls = cd_mthd.get_method()
    kwargs = dict(truths={}, oracle_Z=False, oracle_k=False, k_max=k_max, vb=2)
    cls.fit(mixture_samples, **kwargs)

    print(f"\tDiscovered Edges, {cd_mthd}:")
    for (i, j) in cls.dag.edges: print(f"\t\t{idx2var_dict[i]}->{idx2var_dict[j]}")

    metrics = cls.get_graph_metrics(true_g)

    out_fl = os.path.join("../../results_paper/res_sachs", f"results_m_{cd_mthd}.tsv")
    os.makedirs(os.path.dirname(out_fl), exist_ok=True)
    write_out_metrs(out_fl, metrics)

    if cd_mthd.value == CD.CausalMixtures.value:
        print(f"\tDiscovered Nodes: {cls.e_n_Z}")
        [pd.DataFrame(np.array(cls.e_Z[zi])).to_csv(
            os.path.join("../../results_paper/res_sachs", f"classes_m_{cd_mthd}_Z{zi}.tsv"))
         for zi in range(3)]
        [np.save(os.path.join("../../results_paper/res_sachs", f"classes_m_{cd_mthd}_Z{zi}"), np.array(cls.e_Z[zi]))
         for zi in range(3)]


def sachs_run_mixtureutigsp_ours(mixture_samples, intv_args_dict, intv_targets, t_Z, idx2var_dict):
    # %% MixtureUTIGSP: discovering targets
    # hyperparameters
    t_intv_args_dict = intv_args_dict.copy()
    gmm_tol = 1000
    num_tgt_prior = 12
    cutoff_drop_ratio_list = [0.01, ]
    gSolver = GaussianMixtureSolver("sachs")
    err, intv_args_dict, weight_precision_error, est_num_comp, gm_score_dict, gm \
        = gSolver.mixture_disentangler(num_tgt_prior, intv_args_dict, mixture_samples, gmm_tol,
                                       cutoff_drop_ratio_list[0], )
    mi_scores_ut = sachs_eval_iv_targets(
        [gm.predict(mixture_samples) for _ in range(mixture_samples.shape[1])],
        t_Z, intv_targets, idx2var_dict, t_intv_args_dict)

    out_fl = os.path.join("../../results_paper/res_sachs", f"results_ivtargets_m_{CD.MixtureUTIGSP}.tsv")
    write_out_metrs(out_fl, mi_scores_ut)

    # %% MixtureUTIGSP: discovering G
    est_dag, intv_args_dict, oracle_est_dag, igsp_est_dag, intv_base_est_dag \
        = gSolver.identify_intervention_utigsp(intv_args_dict, mixture_samples.shape[0])
    est_tgts = [node_i for node_i in range(mixture_samples.shape[1]) if
                any(["est_tgt" in intv_args_dict[ky] and node_i in intv_args_dict[ky]["est_tgt"] and ky != "obs" for ky
                     in intv_args_dict])]
    est_lmg = causaldag_to_lmg(est_dag)
    true_lmg = nxdigraph_to_lmg(true_g)
    metrics_ut = compare_lmg_CPDAG(true_lmg, est_lmg)

    out_fl = os.path.join("../../results_paper/res_sachs", f"results_m_{CD.MixtureUTIGSP}.tsv")
    write_out_metrs(out_fl, metrics_ut)

    # %% Ours: discovering targets under true G
    from src.mixtures.topological_causal_mixture import TopologicalCausalMixture
    top = TopologicalCausalMixture(k_max=5)
    top.fit_Z_given_G(mixture_samples, A.T)
    mi_scores_oracle = sachs_eval_iv_targets(top.e_Z_n, t_Z, intv_targets, idx2var_dict, t_intv_args_dict)
    out_fl = os.path.join("../../results_paper/res_sachs", f"results_ivtargets_m_{CD.CausalMixtures}_trueG.tsv")
    write_out_metrs(out_fl, mi_scores_oracle)

    # %% Ours: discovering G and targets
    topEst = TopologicalCausalMixture(k_max=5, extra_refinement=False)
    topEst.fit_graph_and_mixtures(mixture_samples)

    mi_scores_ours = sachs_eval_iv_targets(topEst.e_Z_n, t_Z, intv_targets, idx2var_dict, t_intv_args_dict)
    write_out_metrs(os.path.join("../../results_paper/res_sachs", f"results_ivtargets_m_{CD.CausalMixtures}.tsv"), mi_scores_ours)

    our_lmg = nxdigraph_to_lmg(topEst.topic_graph)
    metrics_ours = compare_lmg_DAG(true_lmg, our_lmg)
    write_out_metrs(os.path.join("../../results_paper/res_sachs", f"results_m_{CD.CausalMixtures}.tsv"), metrics_ours)

    # %% Ours-variant: discovering G and targets using GES with the latent-aware BIC
    gesMix = CD.CausalMixturesGES.get_method()
    gesMix.fit(mixture_samples, k_max=5)

    mi_scores_ges = sachs_eval_iv_targets(gesMix.e_Z_n, t_Z, intv_targets, idx2var_dict, t_intv_args_dict)
    write_out_metrs( os.path.join("../../results_paper/res_sachs", f"results_ivtargets_m_{CD.CausalMixturesGES}.tsv"), mi_scores_ges)

    metrics_ours_ges = compare_lmg_CPDAG(true_lmg,  gesMix.lmg )
    write_out_metrs(os.path.join("../../results_paper/res_sachs", f"results_m_{CD.CausalMixturesGES}.tsv"), metrics_ours_ges)

    # %% GES: discovering G using the latent-unaware BIC
    ges = CD.GES.get_method()
    ges.fit(mixture_samples)

    metrics_ges = compare_lmg_CPDAG(true_lmg, ges.lmg)
    write_out_metrs(os.path.join("../../results_paper/res_sachs", f"results_m_{CD.GES}.tsv"), metrics_ges)

    # some prints
    for (i, j) in true_g.edges: print(f"\t\t{idx2var_dict[i]}->{idx2var_dict[j]}")
    for (i, j) in topEst.topic_graph.edges: print(f"\t\t{idx2var_dict[i]}->{idx2var_dict[j]}")



if __name__ == "__main__":
    t_Z, t_n_Z, intv_args_dict, mixture_samples, num_nodes, A, idx2var_dict = generate_mixture_sachs()
    true_g = nx.from_numpy_array(A.T, create_using=nx.DiGraph)
    intv_targets = [("Akt", 3), ("PKC", 4), ("PIP2", 5), ("Mek", 6), ("PIP3", 7)]

    #%% Table 1, 2
    sachs_run_mixtureutigsp_ours(mixture_samples, intv_args_dict, intv_targets, t_Z, idx2var_dict)

    #%%  Causal Discovery
    KMAX = 5
    CD_ALGOS = [cd for cd in CD]
    for cd_method in CD_ALGOS:
        sachs_run_causal_discovery(mixture_samples, cd_method, KMAX)
