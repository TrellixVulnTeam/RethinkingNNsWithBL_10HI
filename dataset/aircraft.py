import os
import pdb
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

FILENAME_LENGTH = 7
import os

class AircraftDataset(Dataset):
    """
    # Description:
        Dataset for retrieving FGVC Aircraft images and labels
    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image
        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset
        __len__(self):                  returns the length of dataset
    """

    def __init__(self, transform, phase='train', resize=[500,500] , DATAPATH = './data.deepai.org'):
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.resize = resize
        self.DATAPATH = DATAPATH

        if os.path.exists(f'{DATAPATH}/fgvc-aircraft-2013b'):
          pass
        else:
          os.system( f'wget https://data.deepai.org/FGVCAircraft.zip -p ./data.deepai.org/' )
          os.system( f'unzip {DATAPATH}/FGVCAircraft.zip -d {DATAPATH}')

        self.DATAPATH = os.path.join(self.DATAPATH, "fgvc-aircraft-2013b/data")
        variants_dict = {}
        with open(os.path.join(self.DATAPATH, 'variants.txt'), 'r') as f:
            for idx, line in enumerate(f.readlines()):
                variants_dict[line.strip()] = idx
        self.num_classes = len(variants_dict)

        if phase == 'train':
            list_path = os.path.join(self.DATAPATH, 'images_variant_trainval.txt')
        else:
            list_path = os.path.join(self.DATAPATH, 'images_variant_test.txt')

        self.images = []
        self.labels = []
        with open(list_path, 'r') as f:
            for line in f.readlines():
                fname_and_variant = line.strip()
                self.images.append(fname_and_variant[:FILENAME_LENGTH])
                self.labels.append(variants_dict[fname_and_variant[FILENAME_LENGTH + 1:]])

        # transform
        self.transform = transform
        self.stats(phase)
        
    def __getitem__(self, item):
        # image
        image = Image.open(os.path.join(self.DATAPATH, 'images', '%s.jpg' % self.images[item])).convert('RGB')  # (C, H, W)
        image = self.transform(image)

        # return image and label
        return image, self.labels[item]  # count begin from zero

    def __len__(self):
        return len(self.images)

    def stats(self, phase):	
        print(f"{phase}: %d samples spanning %d classes"%(len(self.labels), len(set(self.labels))))
