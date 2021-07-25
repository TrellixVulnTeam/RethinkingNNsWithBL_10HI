import json
import numpy as np
from torch.nn import functional as F
import torch
from argparse import ArgumentParser
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
from weight_hist import benford_r2_model, init_params, init_params_bad
# from genetic_alg import init_params_genetic
from naive_alg import init_params_naive
from models import mobilenet_v3_large, alexnet
# from torchvision.models import mobilenet_v3_large
import os
from functools import partial
import time

def scaleMLH(mlh):
    return (mlh - 0.946244962) / (0.999919115355 - 0.946244962)

class Experiment(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # makes self.hparams under the hood and saves to ckpt
        self.save_hyperparameters()

        # networks
        if self.hparams.dataset == 'cifar10':
            self.model = alexnet(num_classes=10)
        elif self.hparams.dataset == "flowers":
            self.model = alexnet(num_classes=102)
        elif self.hparams.dataset == "dogs":
            self.model = alexnet(num_classes=120)
        elif self.hparams.dataset == "aircraft":
            self.model = alexnet(num_classes=100)
        elif self.hparams.dataset == "fashion":
            self.model = alexnet(num_classes=10, in_channels=1)
        else: # MNIST
            self.model = alexnet(num_classes=10, in_channels=1)

        self.step_count = 0
        self.score_list = []
        self.val_acc_step = []
        self.val_loss_step = []
        self.score_val_end = []
        self.train_acc_step = []
        self.train_loss_step = []

        self.pseudo_patience = 5
        self.max_acc = float('-inf')
        self.earlystopping_counter = 0
        self.earlystopping_marker = None
        self.earlystopping_mlh = None

        self.interval = 1 if self.hparams.stopping_criterion == "none" else 8

        self.window_length = 256#(train_length // self.hparams.batch_size + 1)
        self.train_acc_window = [0.0] * self.window_length

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)

        acc = accuracy_score(y.cpu(), y_pred.cpu().argmax(1).detach())

        # print(self.score_list)

        self.train_acc_step.append(
            {"training_acc": acc, "step": self.step_count})
        self.train_loss_step.append({"training_loss": loss, "step": self.step_count})
        self.train_acc_window.append(acc)
        if len(self.train_acc_window) > self.window_length:
            self.train_acc_window = self.train_acc_window[:-self.window_length]
        if self.step_count % self.interval == 0:
            mlh = benford_r2_model(self.model)
            energy = mlh# + sum(self.train_acc_window) / len(self.train_acc_window)
            self.score_list.append(mlh)
            self.log("energy", energy)
            self.log("mlh/train", mlh, prog_bar=True)
        self.log("loss/train", loss)
        self.log("acc/train", acc, prog_bar=True)
        self.step_count += 1
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)

        acc = accuracy_score(y.cpu(), y_pred.cpu().argmax(1).detach())

        self.log("loss/val", loss, on_epoch=True)
        self.log("acc/val", acc, on_epoch=True)

        return loss.item(), acc

    def validation_epoch_end(self, out):
        loss = np.array([i[0] for i in out]).mean()
        acc = np.array([i[1] for i in out]).mean()
        self.val_acc_step.append(
            {"validation_accuracy": acc, "step": self.step_count})
        self.val_loss_step.append(
            {"validation_loss": loss, "step": self.step_count})
        if self.max_acc <= acc:
            self.max_acc = acc

        if self.earlystopping_counter == self.pseudo_patience:
            self.earlystopping_marker = self.step_count
            self.earlystopping_mlh = acc
            self.earlystopping_counter += 1
        elif self.earlystopping_counter < self.pseudo_patience:
            self.earlystopping_counter += 1

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)

        acc = accuracy_score(y.cpu(), y_pred.argmax(1).cpu().detach())
        return {"loss": loss.item(), "acc": acc}

    def test_epoch_end(self, step_outputs):
        avg_acc = 0
        avg_loss = 0
        for step in step_outputs:
            avg_acc += step["acc"]
            avg_loss += step["loss"]
        self.test_results = {
            "acc/test": avg_acc / len(step_outputs),
            "loss/test": avg_loss / len(step_outputs),
        }

    def configure_optimizers(self):
        lr = self.hparams.learning_rate

        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=50000)
        return [opt], [schedule]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--learning_rate", type=float, default=0.003, help="adam: learning rate"
        )
        parser.add_argument(
            "--adam_b1",
            type=float,
            default=0.9,
            help="adam: decay of first order momentum of gradient",
        )
        parser.add_argument(
            "--adam_b2",
            type=float,
            default=0.999,
            help="adam: decay of first order momentum of gradient",
        )
        parser.add_argument(
            "--initializer", type=str, required=True, help="naive, genetic, glorot"
        )
        parser.add_argument(
            "--experiment_name",
            type=str,
            required=True,
            help="benford_vs_acc, comparison, benford_vs_time, val_acc_vs_time",
        )
        return parser


def cli_main(args=None):
    from pl_bolts.datamodules import CIFAR10DataModule, MNISTDataModule, FashionMNISTDataModule
    from caltech256 import CalTech256DataModule
    from dataset import FlowersDataModule, DogsDataModule, AircraftDataModule

    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str,
                        help="mnist, cifar10, fashion, aircraft, dogs, flowers")
    parser.add_argument("--stopping_criterion", required=True, choices=["energy", "none", "loss/val"])
    parser.add_argument("--val_proportion", type=float)
    script_args, _ = parser.parse_known_args(args)


    if script_args.dataset == "mnist":
        dm_cls = MNISTDataModule
        custom = False
    elif script_args.dataset == "cifar10":
        dm_cls = CIFAR10DataModule
        custom = False
    elif script_args.dataset == "fashion":
        dm_cls = FashionMNISTDataModule
        custom = False
    elif script_args.dataset == 'caltech256':
        dm_cls = CalTech256DataModule
        custom = True
    elif script_args.dataset == 'flowers':
        dm_cls = FlowersDataModule
        custom = True
    elif script_args.dataset == 'dogs':
        dm_cls = DogsDataModule
        custom = True
    elif script_args.dataset == 'aircraft':
        dm_cls = AircraftDataModule
        custom = True

    parser = dm_cls.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Experiment.add_model_specific_args(parser)
    args, _ = parser.parse_known_args(args)

    if args.initializer == "naive":
        init_func = init_params_naive
    elif args.initializer == 'genetic':
        raise ValueError("genetic alg not implemented")
        init_func = init_params_genetic
    elif args.initializer == 'glorot':
        init_func = init_params
    else:
        if args.initializer == "bad":
            init_func = init_params_bad
        else:
            init_func = partial(init_params, initializer=args.initializer)

    if args.stopping_criterion == "energy":
        val_split = 1
        callbacks = [pl.callbacks.EarlyStopping(
            "mlh/train", patience=8, mode="max")]
    elif args.stopping_criterion == "loss/val":
        callbacks = [pl.callbacks.EarlyStopping(
            "loss/val", patience=5, mode="min")]
        val_split = args.val_proportion
    else:
        val_split = 0.3
        callbacks = None

    if custom:
        dm = dm_cls.from_argparse_args(args)
        dm.init(val_split)
    else:
        dm = dm_cls(val_split=val_split, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)

    # if args.stopping_criterion == "energy":
    #     callbacks = [pl.callbacks.EarlyStopping(
    #         "energy", patience=15, mode="max", min_delta=0.1)]

    # if custom:
    #     dm.init(val_split):

    model = Experiment(**vars(args))

    benford_initial = benford_r2_model(model.model)


    trainer = pl.Trainer.from_argparse_args(
        args, max_epochs = 200, callbacks=callbacks,
        precision=16#, accumulate_grad_batches=2
    )
    dm.setup()
    dm.test_dataloader()
    trainer.fit(model, dm)
    trainer.test(model, dm.test_dataloader())

    benford_final = benford_r2_model(model.model)
    training_steps = model.step_count
    final_acc = model.test_results["acc/test"]

    benford_list = model.score_list
    val_acc_step = model.val_acc_step
    train_acc_step = model.train_acc_step
    val_loss_step = model.val_loss_step
    train_loss_step = model.train_loss_step

    out_dict = {
        "initializer": args.initializer,
        "dataset": args.dataset,
        "initial_r2": benford_initial,
        "final_r2": benford_final,
        "training_steps": training_steps,
        "test_acc": final_acc,
        "val_proportion": args.val_proportion,
        "benford_list": benford_list,
        "val_acc_step": val_acc_step,
        "train_acc_step": train_acc_step,
        "val_loss_step": val_loss_step,
        "train_loss_step": train_loss_step,
        "earlystopping_step": model.earlystopping_marker,
        "earlystopping_mlh": model.earlystopping_mlh,
        "stopping_criterion": args.stopping_criterion
    }
    os.makedirs(os.path.join("./stats/", args.experiment_name), exist_ok=True)
    return out_dict, args.experiment_name


def get_idx(experiment_name):
    # file_names = os.listdir(f"./stats/{experiment_name}/")
    # idx = len(file_names)
    return int(time.time())


def main():
    out_dict, experiment_name = cli_main()
    idx = get_idx(experiment_name)
    json.dump(out_dict, open(f"./stats/{experiment_name}/{idx}.json", "w"))
    print(f"dumped {idx}")


if __name__ == "__main__":
    main()
