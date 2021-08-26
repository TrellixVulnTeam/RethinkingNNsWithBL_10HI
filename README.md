# RethinkingNNsWithBL

> Code for the paper "Rethinking Neural Networks with Benford's Law"

`experiments.py`

- contains PyTorch code for conducting all of the experiments in the paper (except for synthetic datasets).
  `run_experiments.py`
- is a python script to run multiple "experiments" in parallel.
- Run `python3 run_experiments.py` to reproduce results for most of the experiments presented in the paper.

`reg_synth.zip`

- contains a similar codebase for generating `Synthetic-Uniform` Dataset from our paper, and experiments related to it.
- extract the zip and run `python3 run_experiments.py` to reproduce results.

`cla_synth.zip`

- contains a similar codebase for generating `Synthetic-Boolean` Dataset from our paper, and experiments related to it.
- extract the zip and run `python3 run_experiments.py` to reproduce results.

`weight_hist.py`

- contains code for computing `MLH` score defined in the paper.
- Initilization method definitions.
- Plotting Layerwise `MLH` for various models.

`models.py`

- contains model definitions for various experiments.
- Info on where each model is used is described in the paper.
