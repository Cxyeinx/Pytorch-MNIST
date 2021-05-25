import torchvision.transforms as transforms
from custom_dataset import Custom_Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import random_split


dataset = Custom_Dataset(root_dir="./archive")
train_set, test_set = random_split(dataset, [round(len(dataset)*0.8), round(len(dataset)*0.2)])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=True)