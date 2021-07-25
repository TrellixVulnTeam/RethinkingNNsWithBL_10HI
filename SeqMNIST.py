import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl


class SeqMNIST_(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        x = x.reshape(x.shape[1] * x.shape[2], 1)
        # print(x.shape)
        return x, y


class SeqMNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 64, num_workers: int = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ds = SeqMNIST_(MNIST("./", download=True, transform=ToTensor()))
        self.train_ds, self.val_ds = torch.utils.data.random_split(
            ds, [len(ds) - 5000, 5000])
        self.test_ds = SeqMNIST_(
            MNIST("./", download=False, transform=ToTensor()))

        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


if __name__ == "__main__":
    dm = SeqMNISTDataModule()
    for x, y in dm.train_dataloader():
        print(x.shape, y.shape)
        break
