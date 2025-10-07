 
### Setup 
- with Docker: ``docker build -t causmix_env .``
- with conda: ``./scripts/setup_env.sh``
- with pip: ``pip install -r requirements.txt``

### Demos
Here are some small usage examples,

- to generate synthetic data and show the fitted mixtures of regressions: [demo_mixing.ipynb](src/examples/demo_mixing.ipynb) 
- to discover causal graphs given synthetic data: [demo_causaldiscovery.ipynb](src/examples/demo_causaldiscovery.ipynb) 
- to run it on your own dataset: [demo_usage.ipynb](src/examples/demo_usage.ipynb)


### Reproducing Figures
The experiment data shown in the main figures is included in [results_paper](results_paper) and can be plotted using the following notebooks,

- Fig. 4: [reproduce_fig_4.ipynb](src/examples/reproduce_fig_4.ipynb) 
- Fig. 5: [reproduce_fig_5.ipynb](src/examples/reproduce_fig_5.ipynb)
- Tb 1 & 2: [reproduce_sachs.ipynb](src/examples/reproduce_sachs.ipynb)

### Reproducing Experiments
To re-run the experiments from scratch, use 

- Fig. 4:
``./scripts/run_fig4.sh`` or ``python run.py -e 0 ``
- Fig. 5: ``./scripts/run_fig5.sh`` or ``python run.py -e 2 ``
- Tb. 1 & 2:  ``./scripts/run_sachs.sh`` or ``python src/exp/run_sachs.py``
- Figs D1, D2: ``./scripts/run_figD1.sh``, ``./scripts/run_figD2.sh``
- Fig. 1, mixing for specific graph structures, e.g., bivariate: ``-e 3``, Fig. 6, interventional mixture: ``-e 4`` 

Useful flags include ``--demo`` to run fewer repetitions, ``--only_base_params`` to run experiments only for the basic parameter setting and not for all parameter combos, ``-wd res/`` to specify the write directory for results and log files.

### References 
The following source code and datasets are included to allow reproducing the experiments,
#### Baselines
- [mixture_mec](src/baselines/mixture_mec):  Kumar, A., Shiragur, K., and Uhler, C., “Learning Mixtures of Unknown Causal Interventions”, <i>arXiv e-prints</i>, Art. no. arXiv:2411.00213, 2024. doi:10.48550/arXiv.2411.00213.
- [sep_distances](src/baselines/sep_distances): Wahl, J. and Runge, J., “Separation-based distance measures for causal graphs”, <i>arXiv e-prints</i>, Art. no. arXiv:2402.04952, 2024. doi:10.48550/arXiv.2402.04952.
- [dodiscover](src/baselines/dodiscover): Python package for representing causal graphs, https://www.pywhy.org/dodiscover/dev/index.html
- [CausalDisco](src/baselines/CausalDisco): baseline algorithms and analytics tools for Causal Discovery in Python, https://github.com/CausalDisco/CausalDisco

#### Datasets
- [sachs_yuhaow.csv](src/dsets/sachs_yuhaow.csv): Karen Sachs et al., Causal Protein-Signaling Networks Derived from Multiparameter Single-Cell Data. Science308,523-529(2005).DOI:10.1126/science.1105809