from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

from torchvision import transforms

class BasicDataset(Dataset):
    def __init__(self, imgs_dir):
        self.imgs_dir   = imgs_dir
        #self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
        #            if not file.startswith('.')]
        self.ids = [imgs_dir]

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):

        transformation = transforms.Compose([transforms.Resize((128,128)),
                                     transforms.ToTensor()])
        img_tr = transformation(pil_img)
        return img_tr

    def __getitem__(self, i):
        #idx = self.ids[i]
        #img_file = glob(self.imgs_dir + idx + '*')
        img_file = self.imgs_dir
        #print(img_file)

        #assert len(img_file) == 1, \
        #    f'Either no image or multiple images found for the ID {idx}: {img_file}'
        #img = Image.open(img_file[0])
        img = Image.open(img_file)
        img = self.preprocess(img)

        return img
