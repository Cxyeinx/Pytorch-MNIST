import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import warnings


class Custom_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        no = len(os.listdir(os.path.join(self.root_dir, "no")))
        yes = len(os.listdir(os.path.join(self.root_dir, "yes")))
        return no + yes

    def __getitem__(self, idx):
        if idx < 0:
            return False

        no = [f"{os.getcwd()}/{self.root_dir}/no/{i}" for i in os.listdir(os.path.join(self.root_dir, "no"))]
        yes = [f"{os.getcwd()}/{self.root_dir}/yes/{i}" for i in os.listdir(os.path.join(self.root_dir, "yes"))]
        lst = no + yes

        if idx > len(lst):
            return False

        label = 0 if idx <= len(no) else 1
        img = Image.open(lst[idx]).resize((256, 256)).convert("L")

        if self.transform:
            img = self.transform(img)
        else:
            transform = transforms.ToTensor()
            img = transform(img)

        return img, label
