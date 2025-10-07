#!/bin/bash
 
set -e

echo "setup of a conda environment to run causal mixtures"
ENV_NAME="cmix"
conda create -n "$ENV_NAME" python=3.10 -y
source activate "$ENV_NAME"
pip install jupyter 
pip install ipykernel
pip install -r requirements.txt
conda install -c r rpy2
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python 3 ($ENV_NAME)"
