from joblib import Parallel, delayed
from numpy.random import SeedSequence

from src.exp.algos import OracleType
from src.exp.config import ExpType
from src.exp.run_cd import run_case_causal_discovery, run_case_interventional_mixture
from src.exp.run_clustering import run_case_clustering
from src.exp.run_power_specificity import run_case_power_specificity
from src.exp.util.case_results import CaseReslts, write_cases
from src.exp.util.run_util import run_info
from src.mixtures.mixing.mixing import MixingType


def run_case_safe(options, params, case, exp, rep_seed, seed_seq):
    try:
        return _run_case(options=options, params=params, case=case, exp=exp, rep_seed=rep_seed, sub_seed=1)
    except Exception as e:
        options.logger.info(f"Error occured {e}, retry")
        cs = seed_seq.spawn(100)
        for sub_seed in enumerate(cs):
            options.logger.info(f"Repeat with seed={sub_seed}")
            try:
                return _run_case(options=options, params=params, case=case, exp=exp, rep_seed=rep_seed,
                                 sub_seed=sub_seed)
            except Exception:
                continue
    try:
        return _run_case(options=options, params=params, case=case, exp=exp, rep_seed=rep_seed, sub_seed=1)
    except Exception as e:
        options.logger.info(f"Error occured {e}")
        raise e


def _run_case(**kwargs):
    """ Run one repetition of a experiment depending on exp type """

    options = kwargs["options"]
    if options.exp_type in [ExpType.CLUSTERING, ExpType.CLUSTERING_GRAPHSTRUCTURES]:
        result = run_case_clustering(**kwargs)
    elif options.exp_type == ExpType.POWER_SPECIFICITY:
        result = run_case_power_specificity(**kwargs)
    elif options.exp_type == ExpType.CAUSAL_DISCOVERY:
        result = run_case_causal_discovery(**kwargs)
    elif options.exp_type == ExpType.INTERVENTIONAL_MIXTURE:
        result = run_case_interventional_mixture(**kwargs)
    else:
        raise ValueError
    # run_info(f'\tFinished Case: {case} for methods {result.keys()}', options.logger, options.verbosity)
    return result


def run_cases(options):
    exps = options.get_experiments()
    cases = options.get_cases()
    reslts = [CaseReslts(case) for case in cases]

    # writing results only
    if options.read:
        for exp in exps:
            for attr in options.fixed:
                write_cases(options, exp, attr, options.read_dir)
        return

    # Each experiment configuration (eg linear Gaussian)
    for exp in exps:

        run_info("", options.logger, options.verbosity)
        run_info("*** Experiment Info ***", options.logger, options.verbosity)
        run_info(f"Experiment Type: {options.exp_type.long_nm()}", options.logger, options.verbosity)
        run_info(f"Base case: {options.get_base_attribute_idf()}", options.logger, options.verbosity)
        nln = '\n\t'
        run_info(f'All cases:\n\t{nln.join(cases)}', options.logger, options.verbosity)
        run_info(f"All CD methods: {', '.join([str(m) for m in options.methods])}", options.logger, options.verbosity)
        run_info(f"All clustering methods: {', '.join(['our MLR' if m==MixingType.MIX_LIN else str(m) for m in options.get_mixing_algos()])}", options.logger, options.verbosity)
        run_info(f"Oracles, if any: {', '.join([ 'without oracles' if m==OracleType.hatGhatZ else str(m) for m in options.get_oracles()])}", options.logger, options.verbosity)
        run_info("", options.logger, options.verbosity)

        # Each parameter configuration (eg N=10 nodes, Z=2 cfds, ...)
        for case, res in zip(cases, reslts):
            run_info(f"CASE: {case}", options.logger, options.verbosity)

            ss = SeedSequence(options.seed)
            cs = ss.spawn(options.reps)
            params = cases[case]
            run_one_rep = lambda rep: (
                run_case_safe(options, params, case, exp, rep, ss) if options.safe
                else _run_case(options=options, params=params, case=case, exp=exp, rep_seed=rep, sub_seed=1))

            if options.n_jobs > 1:
                results = Parallel(n_jobs=options.n_jobs)(delayed(
                    run_one_rep)(rep_seed) for rep_seed in enumerate(cs))
            else:
                results = [run_one_rep(rep_seed) for rep_seed in enumerate(cs)]

            res.add_reps(results)
            res.write_case(params, exp, options)

        for attr in options.fixed:
            write_cases(options, exp, attr)
