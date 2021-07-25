import numpy as np
import plotly.express as px
from functools import partial
from scipy import stats
from joblib import Parallel, delayed

benford = np.array([30.1,
                    17.6,
                    12.5,
                    9.7,
                    7.9,
                    6.7,
                    5.8,
                    5.1,
                    4.6]) / 100

# @njit
def benford_r2(bin_percent):
    return stats.pearsonr(benford, bin_percent[1:])[0]

import torch
# @njit
def bincount(tensor):
    counts = torch.zeros(10)
    for i in range(10):
        counts[i] = torch.count_nonzero(tensor == i)
    return counts


@torch.no_grad()
def bin_percent(tensor):
    tensor = torch.tensor(tensor)
    tensor = tensor.abs() * 1e10
    tensor = tensor // 10 ** torch.log10(tensor).long()
    tensor = bincount(tensor.long())
    return (tensor / tensor.sum()).numpy()
from numba import njit

@njit
def boltzmann_gibbs(beta, E):
    """Returns the probability of an Ideal Gas in a seal container system at temperature T will have energy E

    Args:
        T (float): Temperature of the Ideal Gas
        E (float): Energy of the Ideal Gas

    Returns:
        float: Probability
    """
    return beta * np.exp(-beta * E)

@njit
def fermi_dirac(E, beta):
    """Returns the probability of an Ideal Gas in a seal container system at temperature T will have energy E

    Args:
        T (float): Temperature of the Ideal Gas
        E (float): Energy of the Ideal Gas

    Returns:
        float: Probability
    """
    # print(beta)
    return (beta/np.log(2)) / (1 + np.exp(beta * E))

class FermiDirac(stats.rv_continuous):
    def _pdf(self, E, beta):
        return (beta/np.log(2)) / (1 + np.exp(beta * E))

def get_beta_i(beta_i, num_vals):
    fd = FermiDirac(a=0.0)
    samples = fd.rvs(beta_i, size=num_vals)
    mlh = benford_r2(bin_percent(samples))
    print(mlh)
    return mlh

from tqdm import tqdm
def generate_samples():
    # beta = np.logspace(-0.1, -1, base=0.1, num=10000, endpoint=True)
    beta = np.linspace(0.1, 1.0, num=1000)
    num_vals = int(2**12)

    mlh = Parallel(n_jobs=-1)(delayed(get_beta_i)(b, num_vals) for b in tqdm(beta))
    fig = px.scatter(x=beta, y=mlh)
    fig.show()

def plot_pdf():
    space = np.linspace(0.0, 1000000, num=10000*10)
    probs = [fermi_dirac(1, e) for e in space]
    fig = px.line(x=space, y=probs)
    fig.show()

if __name__ == "__main__":
    generate_samples()
    