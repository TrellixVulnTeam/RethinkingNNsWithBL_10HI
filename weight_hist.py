import torch
import plotly.graph_objects as go
import numpy as np
from torch import nn
from torch.nn import init
from scipy.stats import pearsonr
from copy import deepcopy
import math

benford = np.array([30.1,
                    17.6,
                    12.5,
                    9.7,
                    7.9,
                    6.7,
                    5.8,
                    5.1,
                    4.6]) / 100
benford_th = torch.FloatTensor(benford)


def non_bias(m, include_bn=False):
    if include_bn:
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
            return m.weight
        return None
    else:
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            return m.weight
        return None


def benford_r2(bin_percent):
    return pearsonr(benford, bin_percent[1:])[0]
# def benford_r2(bin_percent):
#     return kl_div(benford, bin_percent.reshape(-1)[1:]).sum()

def bincount(tensor):
    counts = torch.zeros(10)
    for i in range(10):
        counts[i] = torch.count_nonzero(tensor == i)
    return counts


@torch.no_grad()
def bin_percent(tensor):
    tensor = tensor.abs() * 1e10
    tensor = tensor // 10 ** torch.log10(tensor).long()
    tensor = bincount(tensor.long())
    return tensor / tensor.sum()


def block_bincount(net, include_bn=False):
    bins = []
    num_params = []
    total_num_params = 0
    for m in net.modules():
        # Check if leaf module
        if list(m.children()) == []:
            weight = non_bias(m, include_bn=include_bn)
            if weight is not None:
                n_param = weight.numel()
                num_params.append(n_param)
                total_num_params += n_param
                bins.append(bin_percent(weight.view(-1).detach()))

    out = torch.zeros(10)
    for b, n_param in zip(bins, num_params):
        out += (b) * (n_param / total_num_params)
    return out


def benford_r2_model(model, include_bn=False):
    bins = block_bincount(deepcopy(model).cpu(), include_bn=include_bn)
    return benford_r2(bins)


arg_to_init_fn = {
    'kaiming_uniform_': init.kaiming_uniform_,
    'kaiming_normal_': init.kaiming_normal_,
    'xavier_uniform_': init.xavier_uniform_,
    'xavier_normal_': init.xavier_normal_,
    'orthogonal_': init.orthogonal_,
    'normal_': init.normal_,
    'uniform_': init.uniform_
}


def init_params(net, initializer, bias=False):
    '''Init layer parameters.'''
    init_fn = arg_to_init_fn[initializer]
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init_fn(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init_fn(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)


def kaiming_normal_bad(tensor, mean, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = init._calculate_correct_fan(tensor, mode)
    gain = init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(mean, std)


def init_params_bad(net, bias=False):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_normal_bad(m.weight, mean=np.random.uniform(0.1, 0.5))
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, mean=np.random.uniform(0.1, 0.5))
            if m.bias is not None:
                init.constant_(m.bias, 0)


def plot(bin_percent, title=None):
    fig = go.Figure(data=[
        go.Bar(x=np.arange(10)[1:], y=bin_percent[1:], name='Weights'),
        go.Scatter(x=np.arange(10)[1:], y=benford, name="Benford's Law")
    ])
    fig.update_layout(title=title)
    fig.show()


def plot_model_bar(untrained, trained, title, exclude_fc=False, fc_only=False):
    if fc_only:
        p = trained.fc.weight.view(-1).detach()
        bins_tr = bin_percent(p)

        p = untrained.fc.weight.view(-1).detach()
        bins_utr = bin_percent(p)
    else:
        p = deepcopy(trained)
        if exclude_fc:
            p.fc = None
        bins_tr = block_bincount(p)

        p = deepcopy(untrained)
        if exclude_fc:
            p.fc = None
        bins_utr = block_bincount(p)

    fig = go.Figure(data=[
        go.Bar(x=np.arange(10)[1:], y=bins_tr[1:], name='Trained'),
        go.Bar(x=np.arange(10)[1:], y=bins_utr[1:], name='Random'),
        go.Scatter(x=np.arange(10)[1:], y=benford, name="Benford's Law")
    ])
    fig.update_layout(
        title=title,
        barmode='group'
    )

    print(" " * 21 + "Pearson's R v/s Benford's Law")
    print("{:20}".format("Random"),  round(benford_r2(bins_utr), 4))
    print("{:20}".format("Trained"), round(benford_r2(bins_tr), 4))

    # fig.show()
    return fig


def plot_model_layerwise(untrained, trained, title, layer_names=None):
    scores1, scores2 = [], []
    flag = 0
    if layer_names is None:
        layer_names = []
        flag = 1
    i = 1
    for m1, m2 in zip(trained.children(), untrained.children()):
        if sum(p.numel() for p in m1.parameters()) == 0 or isinstance(m1, nn.BatchNorm2d):
            continue
        score1, score2 = benford_r2(block_bincount(
            m1)), benford_r2(block_bincount(m2))
        scores1.append(score1)
        scores2.append(score2)
        if flag:
            layer_names.append(f"Block{i}")
        i += 1
    if flag:
        layer_names[-1] = "FC"

    fig = go.Figure(data=[
        go.Scatter(x=layer_names, y=scores1, name='Trained'),
        go.Scatter(x=layer_names, y=scores2, name='Random'),
    ])
    fig.update_layout(
        title=title,
        barmode='group'
    )

    # fig.show()
    return fig


if __name__ == "__main__":
    layer1 = torch.nn.Sequential(
        torch.nn.Linear(784, 10),
        torch.nn.Linear(784, 10)
    )
    layer2 = torch.nn.Sequential(
        torch.nn.Linear(784, 10),
        torch.nn.Linear(784, 10)
    )
    init_params(layer1, 'kaiming_normal_')
    # f = bin_percent(layer.weight.view(-1))
    # print(f)
    # print(f.shape)

    # plot(f, 'dummy title')

    # plot_model_bar(layer1, layer2, 'mlp')
    # plot_model_layerwise(layer1, layer2, 'layerwise')
