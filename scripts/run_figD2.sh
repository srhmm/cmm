docker run -v $(pwd):/workspace -w /workspace causmix_env /bin/bash -c "source activate causmix && python run.py -e 2 --include_ges_variant --only_base_params --safe"
