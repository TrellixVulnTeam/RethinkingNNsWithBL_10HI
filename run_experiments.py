import multiprocessing as mp
import os
from itertools import product

datasets = ['cifar10'] * 100
val_proportions = [0.1, 0.2, 0.3, 0.4]
val_proportions += [0.00010000000000000014, 0.0003981071705534976,
                    0.0015848931924611156, 0.025118864315095833]


def spawn_worker_exp2(args):
    """ Stopping Criterion is Validation loss """
    val_proportion, dataset = args
    os.system("mkdir ./stats/;")
    os.system(f"mkdir ./stats/{dataset};")
    command = f"python3 experiments.py --dataset={dataset} --val_proportion {val_proportion} --stopping_criterion loss/val --initializer=glorot --gpus=1 --batch_size=512 --val_check_interval=0.5 --experiment_name alexnet_val --num_workers=2"
    os.system(command)


def spawn_worker_exp1(dataset):
    """ Stopping Criterion is MLH """
    os.system("mkdir ./stats/;")
    os.system(f"mkdir ./stats/{dataset};")
    command = f"python3 experiments.py --dataset={dataset} --stopping_criterion energy --initializer=glorot --gpus=1 --batch_size=512 --val_check_interval=1.0 --experiment_name alexnet_energy --num_workers=2"
    os.system(command)


def spawn_worker_exp3(dataset):
    """ No Early Stopping """
    os.system("mkdir ./stats/;")
    os.system(f"mkdir ./stats/{dataset};")
    command = f"python3 experiments.py --dataset={dataset} --stopping_criterion none --initializer=glorot --gpus=1 --batch_size=512 --val_check_interval=0.33 --experiment_name energy_vs_val_alexnet_none_regression --num_workers=2"
    os.system(command)


with mp.Pool(4) as p:
    p.map(spawn_worker_exp1, datasets)
with mp.Pool(4) as p:
    p.map(spawn_worker_exp2, product(val_proportions, datasets))
with mp.Pool(4) as p:
    p.map(spawn_worker_exp3, datasets)
