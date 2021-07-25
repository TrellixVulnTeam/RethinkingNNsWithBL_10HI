import multiprocessing as mp
import os
from itertools import product

# datasets = ['mnist'] * 20
# initializers = ['kaiming_uniform_', 'kaiming_normal_', 'xavier_uniform_',
#                 'xavier_normal_', 'orthogonal_', 'normal_', 'uniform_', 'bad']


# def spawn_worker_exp1(args):
#     dataset, initializer = args
#     os.system("mkdir ./stats/;")
#     os.system("mkdir ./stats/mnist_initializer;")
#     command = f"~/miniconda3/bin/python3 experiments.py --dataset={dataset} --initializer={initializer} --gpus=1 --batch_size=64 --num_workers=2 --val_check_interval=0.33 --experiment_name mnist_initializer"
#     os.system(command)


# with mp.Pool(mp.cpu_count() // 2) as p:
#     p.map(spawn_worker_exp1, product(datasets, initializers))

# datasets = ['fashion', 'mnist', 'cifar10', 'seqmnist'] * 100


# def spawn_worker_exp2(dataset):
#     os.system("mkdir ./stats/;")
#     os.system("mkdir ./stats/datasets;")
#     command = f"python3 experiments.py --dataset={dataset} --initializer=glorot --gpus=1 --batch_size=64 --val_check_interval=0.33 --experiment_name datasets --num_workers=3"
#     os.system(command)


# with mp.Pool(2) as p:
#     p.map(spawn_worker_exp2, datasets)


# datasets = ['fashion', 'mnist', 'cifar10', 'seqmnist'] * 100
# datasets = ['aircraft', 'flowers', 'dogs', 'cifar10', 'fashion'] * 5
# datasets = ['cifar10', 'fashion'] * 20


# def spawn_worker_exp3(dataset):
#     os.system("mkdir ./stats/;")
#     os.system(f"mkdir ./stats/{dataset};")
#     command = f"python3 experiments.py --dataset={dataset} --initializer=glorot --gpus=1 --batch_size=64 --val_check_interval=0.33 --experiment_name {dataset} --num_workers=3"
#     os.system(command)


# with mp.Pool(3) as p:
#     p.map(spawn_worker_exp3, datasets)

datasets = ['cifar10'] * 100
val_proportions = [0.1, 0.2, 0.3, 0.4]
val_proportions += [0.00010000000000000014, 0.0003981071705534976, 0.0015848931924611156, 0.025118864315095833]
# import numpy as np
# val_proportions = list(np.linspace(0.0015848931924611156, 0.1, endpoint=False, num=20))[1:]

def spawn_worker_exp4(args):
    val_proportion, dataset = args
    os.system("mkdir ./stats/;")
    os.system(f"mkdir ./stats/{dataset};")
    command = f"python3 experiments.py --dataset={dataset} --val_proportion {val_proportion} --stopping_criterion loss/val --initializer=glorot --gpus=1 --batch_size=512 --val_check_interval=0.5 --experiment_name alexnet_val --num_workers=2"
    os.system(command)

def spawn_worker_exp5(dataset):
    os.system("mkdir ./stats/;")
    os.system(f"mkdir ./stats/{dataset};")
    command = f"python3 experiments.py --dataset={dataset} --stopping_criterion energy --initializer=glorot --gpus=1 --batch_size=512 --val_check_interval=1.0 --experiment_name alexnet_energy --num_workers=2"
    os.system(command)

def spawn_worker_exp6(dataset):
    os.system("mkdir ./stats/;")
    os.system(f"mkdir ./stats/{dataset};")
    command = f"python3 experiments.py --dataset={dataset} --stopping_criterion none --initializer=glorot --gpus=1 --batch_size=512 --val_check_interval=0.33 --experiment_name energy_vs_val_alexnet_none_regression --num_workers=2"
    os.system(command)

# with mp.Pool(4) as p:
#     p.map(spawn_worker_exp5, datasets)
with mp.Pool(4) as p:
    p.map(spawn_worker_exp4, product(val_proportions, datasets))
# with mp.Pool(4) as p:
#     p.map(spawn_worker_exp6, datasets)
