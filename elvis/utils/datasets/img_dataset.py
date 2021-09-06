import os
import pdb

from PIL import Image
from torchvision.datasets import VisionDataset


class IMGDataset(VisionDataset):
    def __init__(self, img_folder, transform=None) -> None:
        super(IMGDataset, self).__init__(root=img_folder, transforms=transform)
        self.root           = img_folder
        self.indices        = [img_name for img_name in os.listdir(img_folder)]
        self.transform      = transform

 
    def __getitem__(self, idx):
        img_name    = self.indices[idx]
        #convert to RGB since some images are L (black & white)
        img         = Image.open(os.path.join(self.root, img_name)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


    def __len__(self):
        return len(self.indices)
