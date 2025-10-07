import networkx as nx
import numpy as np

from src.exp.algos import CD
from src.exp.gen.generate import GSType
from src.exp.run_power_specificity import compare_graph_metrics
from src.mixtures.mixing.mixing import MixingType, fit_mixture_model
from src.mixtures.topological_causal_mixture import TopologicalCausalMixture
from src.mixtures.util.util import compare_Z




def demo_causal_discovery(X, truths, params, causaldiscovery_method_ty=CD.SCORE, KMAX=5, vb=False, eval=True):
    method_ty = causaldiscovery_method_ty
    results = dict()
    #print(f'Method: {method_ty.value}')
    DISCOVERS_MIXING = method_ty.discovers_mixture_assignments()

    # model fitting
    cls = method_ty.get_method()
    kwargs = dict(truths=truths, oracle_Z=False, k_max=KMAX, vb=2, lambda_mix=0)
    cls.fit(X, **kwargs)

    if not eval: return cls, {}
    true_nxg = truths['true_g']
    # eval
    metrics = cls.get_graph_metrics(true_nxg)

    # Mixture eval
    if DISCOVERS_MIXING:
        assert cls.e_Z_n is not None and cls.e_n_Z is not None
        metrics.update(compare_Z(X.shape[0], truths['t_A'], nx.to_numpy_array(cls.dag),
                                   truths['t_Z'], truths['t_n_Z'], None, None, cls.e_n_Z, cls.e_Z_n, None, GSType.GRAPH))
    print("")
    print(f"\t***\tCD: {method_ty}")
    print(
        f"\tEval graph, \tF1: {metrics['f1']:.2f}, SHD: {metrics['shd']:.2f}, TP: {metrics['tp']} FP: {metrics['fp']}")

    if DISCOVERS_MIXING:
        print(
            f"\tEval Mixing: \tJacc {metrics.get('jacc', -1):.2f} AMI {metrics.get('node-amis', -1):.2f} F1-iv {metrics.get('f1-iv', -1):.2f} \t{[f'{met}: {vl:.2f}' for met, vl in metrics.items()]}",
             )
        print(f"\tEstimated Targets: {cls.e_n_Z}, True: {truths['t_n_Z']}")

    nm = method_ty.value
    #results[nm] = dict(mth=nm, metrics=metrics)
    return cls, metrics

def demo_pruning(X, truths, params, causaldiscovery_method_ty=CD.SCORE, KMAX=5, vb=False):
    method_ty = causaldiscovery_method_ty
    results = dict()
    print(f'Method: {method_ty.value}')
    t_A, t_n_Z, t_Z = nx.to_numpy_array(truths['true_g']), truths['t_n_Z'], truths['t_Z']
    DISCOVERS_MIXING = method_ty.discovers_mixture_assignments()

    # BASELINE causal discovery (for JPCMI etc: add clustering to fit!!)
    cls = method_ty.get_method()
    #mth, oracle = method_ty.get_method(), method_ty.is_oracle()
    kwargs = dict(truths=truths, #args=None, intv_args_dict=None,
                  oracle_Z=False, k_max=KMAX, vb=2, lambda_mix=0)
    cls.fit(X, **kwargs)

    true_nxg = truths['true_g']
    metrics_cd = cls.get_graph_metrics(true_nxg)
    print(
        f"\tCD \tF1: {metrics_cd['f1']:.2f}, SD: {metrics_cd['sd']:.2f}, TP: {metrics_cd['tp']} FP: {metrics_cd['fp']}")


    # OURS causal discovery and pruning
    base_G = cls.dag
    base_A = nx.to_numpy_array(cls.dag)
    oracle_Z = False
    hypparams=dict(truths=truths, oracle_Z=oracle_Z, oracle_K=False, oracle_G=False, k_max=KMAX, vb=0)
    top = TopologicalCausalMixture(**hypparams)
    top.fit_Z_given_G(X, base_G.copy()) # copy as pruning modifies it
    our_G = top.topic_graph
    our_A = nx.to_numpy_array(our_G)

    metrics_base = compare_dag(truths['true_g'], base_G)
    metrics_ours = compare_dag(truths['true_g'], our_G)

    print("")
    print(f"\t***\tBaseline: {method_ty}")
    print(
        f"\tBasic, \tF1: {metrics_base['f1']:.2f}, SHD: {metrics_base['shd']:.2f}, TP: {metrics_base['tp']} FP: {metrics_base['fp']}" )
    print(
            f"\tMixing, \tF1: {metrics_ours['f1']:.2f}, SHD: {metrics_ours['shd']:.2f}, TP: {metrics_ours['tp']} FP: {metrics_ours['fp']}" )


    # TODO idea: eval a second version of pruning via independence/invariance testing (opposed to currently: functional modelling)
    # try out the pval thing used in CAM, SCORE
    #
    metrics_improv, metrics_ours = compare_graph_metrics(t_A, base_A, our_A, metrics_base, metrics_ours, 'ours')

    print( f"\tImprovements: \tSHD {metrics_improv['shd']:.2f}\t{[f'{met}: {vl:.2f}' for met, vl in metrics_improv.items()]}")

    metrics_mixing =  compare_Z(X.shape[0],truths['t_A'], nx.to_numpy_array(top.topic_graph),
                                truths['t_Z'], truths['t_n_Z'],  top.e_Z, top.Z_pairs, top.e_n_Z,top.e_Z_n,  top.pprobas )
    #print( f"\tMixing Eval: \tJacc {metrics_mixing['jacc']:.2f} AMI {metrics_mixing['node_amis']:.2f} \t{[f'{met}: {vl:.2f}' for met, vl in metrics_mixing.items()]}" )
    print(f"\tEstimated Targets: {top.e_n_Z}, True: {truths['t_n_Z']}")

    metrics_ours.update(metrics_mixing)

    nm = method_ty.value
    results[nm+'_base'] = dict(mth=nm+'_base', metrics=metrics_base)
    results[nm+'_ours'] = dict(mth=nm+'_pruned', metrics=metrics_ours)
    results[nm+'_mixing'] = dict(mth=nm+'_our_mixing', metrics=metrics_mixing)
    results[nm+'_improv'] = dict(mth=nm+'_improv_nolatent', metrics=metrics_improv)
    return results
def demo_clustering(data, truths, params, mixing_ty=MixingType.MIX_LIN,
                    causaldiscovery_method_ty=CD.SCORE, ORACLE_G=True, ORACLE_K=False, ORACLE_Z=False, KMAX=5, vb=False, ret_model=False, our_vb=0):
    SKIP_PRUNING = True
    DISCOVER_G = not ORACLE_G
    DISCOVER_MIXING_EACH_NODE = mixing_ty.search_each_node()
    #t_A, t_n_Z, t_Z = nx.to_numpy_array(truths['true_g']), truths['t_n_Z'], truths['t_Z']

    if ORACLE_G:
        given_A = truths['true_g']
    else:
        assert DISCOVER_G
        assert causaldiscovery_method_ty is not None
        mth = causaldiscovery_method_ty.get_method()
        kwargs = dict(mixing_type=mixing_ty, truths=truths, hybrid=False, oracle_Z=ORACLE_Z, oracle_K=ORACLE_K,
                      oracle_G=ORACLE_G,
                      kmax=KMAX)  # todo args for other methods in options
        mth.fit(data, **kwargs)
        given_A = nx.from_numpy_array(mth.get_directed_graph(), create_using=nx.DiGraph)

        # CMMs: our causal mixture discovery
        # (and MMs for each node in turn)
    if DISCOVER_MIXING_EACH_NODE:
        hypparams = dict(
            mixing_type=mixing_ty,
            truths=truths, oracle_Z=ORACLE_Z, oracle_K=ORACLE_K, oracle_G=ORACLE_G, k_max=KMAX, vb=our_vb)
        # truths.dg.plot_X(data)
        ours = TopologicalCausalMixture(**hypparams)

        # MM models: one mixture per node was discovered (ignoring the graph)
        if mixing_ty.is_unconditional_mixture():
            ours.fit_Z_given_G(data, nx.to_numpy_array(given_A.copy()), SKIP_PRUNING,
                               skip_sets=True)  # skip set aggregation using MI stuff which is part of our approach- here only MMs
            e_Z_n = ours.e_Z_n
            # extract "intervention targets": each node that has more than one cluster
            # e_n_Z = [set(nodei) for nodei in range(e_Z_n) if len(np.unique(e_Z_n[nodei]))>1]
            e_n_Z = [set([node for node in range(len(e_Z_n)) if len(np.unique(e_Z_n[node])) > 1])]
            metrics_mixing = compare_Z(data.shape[0], truths['t_A'], None,
                                       truths['t_Z'], truths['t_n_Z'], None, None, e_n_Z, e_Z_n, None, params["GS"])
        else:
            ours.fit_Z_given_G(data, nx.to_numpy_array(given_A.copy()), SKIP_PRUNING)  # copy as pruning modifies it
            e_Z, Z_pairs, e_n_Z, e_Z_n, pprobas = ours.e_Z, ours.Z_pairs, ours.e_n_Z, ours.e_Z_n, ours.pprobas
            metrics_mixing = compare_Z(data.shape[0], truths['t_A'], nx.to_numpy_array(ours.topic_graph),
                                       truths['t_Z'], truths['t_n_Z'], e_Z, Z_pairs, e_n_Z, e_Z_n, pprobas, params["GS"])


    else:
        range_k = range(1, KMAX + 1) if not ORACLE_K else params["K"]
        idl, pproba, _ = fit_mixture_model(mixing_ty, data, range_k)
        e_n_Z = [set([node for node in range(data.shape[1])]) if len(np.unique(idl)) > 1 else set()]
        e_Z_n = [idl for _ in range(data.shape[1])]
        metrics_mixing = compare_Z(data.shape[0], truths['t_A'], None,
                                   truths['t_Z'], truths['t_n_Z'], None, None, e_n_Z,
                                   e_Z_n, [pproba for _ in range(len(data))], params["GS"])
    if vb: print(f"\tResult for {mixing_ty}:\t{metrics_mixing}")
    if vb: print(f"\tEstimated Targets: {e_n_Z}, True: {truths['t_n_Z']}")
    if ret_model: assert ours is not None
    if ret_model: return ours, metrics_mixing
    return e_n_Z, e_Z_n, metrics_mixing
