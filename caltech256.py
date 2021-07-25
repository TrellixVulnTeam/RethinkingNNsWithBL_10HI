from zipfile import ZipFile
import torch
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class CalTech256(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        try:
            self.items = torch.load("./temp_caltech_processed.pkl")
        except:
            self.items = []
            zip = ZipFile("./caltech256.zip", "r")
            # print(self.zip.namelist())
            names = [f for f in zip.namelist() if f.find(".jpg") != -1]
            from tqdm import tqdm
            for idx in tqdm(range(len(names))):
                name = names[idx]
                img = Image.open(zip.open(name)).convert("RGB")
                img = transform(img)
                label = self.get_label(name)
                self.items.append((img, label))
            torch.save(self.items, "./temp_caltech_processed.pkl")

    def __getitem__(self, idx):
        return self.items[idx]

    def __len__(self):
        return len(self.items)

    def get_label(self, file_name):
        return int(file_name.split("/")[1][:3]) - 1


class CalTech256DataModule(pl.LightningDataModule):
    def __init__(
            self,
            num_workers: int = 16,
            batch_size: int = 32,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.num_workers = num_workers
        ds = CalTech256()

        train_examples = int(len(ds) * 0.6)
        test_examples = int(len(ds) * 0.2)
        val_examples = len(ds) - train_examples - test_examples

        self.train, self.val, self.test = torch.utils.data.random_split(
            ds,
            [train_examples, val_examples, test_examples],
            generator=torch.Generator().manual_seed(1234)
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)


if __name__ == "__main__":
    ds = CalTech256()
    dl = torch.utils.data.DataLoader(ds, 64, num_workers=8)
    label_max = -1
    for i, (img, label) in enumerate(dl):
        # print(img.shape, label.shape, i)
        label_max = max(label_max, max(label))
        print(label_max)
