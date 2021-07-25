import os

from torchvision import transforms, datasets
from .stanford_dogs_data import dogs
from .oxford_flowers import flowers
from os.path import join, expanduser
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from .aircraft import AircraftDataset
import torch

root = expanduser("")
imagesets = join(root, 'DATASETS', 'IMAGE')
videosets = join(root, 'DATASETS', 'VIDEO')
models = join(root, 'Models')
plots = join(root, 'Plots')

class DataModule(pl.LightningDataModule):
    def __init__(self, 
                    dataset: str,
                    num_workers=8,
                    batch_size=32,
                    image_size=128,
        ):
        super().__init__()

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.image_size = image_size
        self.dataset = dataset

    def init(self, val_split):
        self.train, self.valid, self.test, classes = load_datasets(self.dataset, input_size=(self.image_size, self.image_size), val_split=val_split)
        self.num_classes = len(classes)
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=False)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False)

    @property
    def num_samples(self):
        return len(self.train)
        
class DogsDataModule(DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__("stanford_dogs", *args, **kwargs)
    
class FlowersDataModule(DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__("oxford_flowers", *args, **kwargs)
    
class AircraftDataModule(DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__("aircraft", *args, **kwargs)

def load_datasets(set_name, input_size=224, val_split=0.3):
    input_transforms = transforms.Compose([
            # transforms.RandomResizedCrop(input_size),
            transforms.Resize(input_size),
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        
    test_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor()])
    if set_name == 'mnist':
        train_dataset = datasets.MNIST(root=os.path.join(imagesets, 'MNIST'),
                                                   train=True,
                                                   transform=transforms.ToTensor(),
                                                   download=True)
        test_dataset = datasets.MNIST(root=os.path.join(imagesets, 'MNIST'),
                                                  train=False,
                                                  transform=transforms.ToTensor())

    elif set_name == 'stanford_dogs':
        train_dataset = dogs(root=imagesets,
                                 train=True,
                                 cropped=False,
                                 transform=input_transforms,
                                 download=True)
        test_dataset = dogs(root=imagesets,
                                train=False,
                                cropped=False,
                                transform=test_transform,
                                download=True)

        classes = train_dataset.classes

        print("Training set stats:")
        train_dataset.stats()
        print("Testing set stats:")
        test_dataset.stats()

    elif set_name == 'oxford_flowers':
        train_dataset = flowers(root=imagesets,
                                  train=True,
                                  val=False,
                                  transform=input_transforms,
                                  download=True)
        test_dataset = flowers(root=imagesets,
                                train=False,
                                val=True,
                                transform=test_transform,
                                download=True)
        classes = list(range(102))
        print("Training set stats:")
        train_dataset.stats()
        print("Testing set stats:")
        test_dataset.stats()

    elif set_name == 'aircraft':
        train_dataset, test_dataset = AircraftDataset(phase='train', transform=input_transforms), AircraftDataset(phase='val', transform=test_transform)
        classes = list(range(train_dataset.num_classes))
    else:
        return None, None

    total = len(train_dataset)
    if isinstance(val_split, int):
        valid_length = val_split
    else:
        valid_length = int(total * val_split)
    train_length = total - valid_length
    
    lengths = [train_length, valid_length]
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, lengths) 
    return train_dataset, val_dataset, test_dataset, classes

AircraftDataModule()
FlowersDataModule()
DogsDataModule()
