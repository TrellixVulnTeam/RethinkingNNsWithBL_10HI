# RethinkingNNsWithBL

> Code for the paper "Rethinking Neural Networks with Benford's Law" to appear in NeurIPS 2021 Machine Learning for Physical Sciences Workshop.

## Usage

- To reproduce Experiment in Table 2 and Fig. 4, run `python3 run_experiments.py`. This will train over 900 LeNet-like models, and will run for a very long time. The results would be collected as `json` files at `./stats/`. Tensorboard logs will be generated at `lightning_logs`. We have provided experimental data at `stats_fig4` for our run.

- Plots in Fig. 3 were plotted using `early stopping results.ipynb`

- Plots in Fig. 5 were plotted using `plot_simulation.ipynb`.

## File Descriptions

`experiments.py`

- contains PyTorch code for conducting all of the experiments in the paper (except for synthetic datasets).
  `run_experiments.py`
- is a python script to run multiple "experiments" in parallel.
- Run `python3 run_experiments.py` to reproduce results for most of the experiments presented in the paper.

`weight_hist.py`

- contains code for computing `MLH` score defined in the paper.
- Initilization method definitions.
- Plotting Layerwise `MLH` for various models.

`models.py`

- contains model definitions for various experiments.
- Info on where each model is used is described in the paper.
