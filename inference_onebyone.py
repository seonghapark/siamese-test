import numpy as np
import random
from PIL import Image
import PIL.ImageOps

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

from siamese_net import SiameseNetwork
from inference_dataset import BasicDataset

from pathlib import Path
import argparse
import os
import glob

class Main():
    def __init__(self, args):
        self.gpu = torch.device(args.device)
        self.model = SiameseNetwork().cuda(self.gpu)

        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def run(self, args, train_loader, val_loader):
        self.model.eval()
        for step, (x, y) in enumerate(zip(train_loader, val_loader), len(val_loader)):
            x = x.cuda(self.gpu, non_blocking=True)
            y = y.cuda(self.gpu, non_blocking=True)

            output1, output2 = self.model(x, y)
            euclidean_distance = F.pairwise_distance(output1, output2)

        return euclidean_distance.item()

def same_folder(args, files):
    for i in range(len(files)):
        if i != len(files)-1:
            args.data_dir1 = files[i]
            args.data_dir2 = files[i+1]
        else:
            args.data_dir1 = files[i]
            args.data_dir2 = files[i-1]

        train_dataset = BasicDataset(args.data_dir1)
        val_dataset = BasicDataset(args.data_dir2)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

        values = []
        val = np.asarray(main.run(args, train_loader, val_loader))
        values.append(val)

    return values


def diff_folders(args, f1, f2):
    if len(f1) <= len(f2):
        files1 = f1
        files2 = f2
    else:
        files1 = f2
        files2 = f1

    for i in range(len(files1)):
        args.data_dir1 = files1[i]
        args.data_dir2 = files2[i]

        train_dataset = BasicDataset(args.data_dir1)
        val_dataset = BasicDataset(args.data_dir2)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

        values = []
        val = np.asarray(main.run(args, train_loader, val_loader))
        values.append(val)

    return values


def with_the_folders(args):
    name = args.data_dir1 + '*'
    files1 = sorted(glob.glob(name))
    name = args.data_dir2 + '*'
    files2 = sorted(glob.glob(name))

    if args.data_dir1 == args.data_dir2:
        loader1, loader2 = same_folder(args, files1)
    else:
        loader1, loader2 = diff_folders(args, files1, files2)

    return loader1, loader2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with VICReg", add_help=False)
    parser.add_argument("--data-dir", type=str, default="/path/to/imagenet", required=True,
                        help='Path to the image net dataset')
    parser.add_argument("--exp-dir", type=Path, default="./exp",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    args = parser.parse_args()
    main = Main(args)

    folders = sorted(os.listdir(args.data_dir))

    limit = 50

    with open('dissimilarity.txt', 'w') as f:
        for i in folders:
            for j in folders:
                if int(i) < limit and int(j) < limit:
                    args.data_dir1 = args.data_dir + i + '/'
                    args.data_dir2 = args.data_dir + j + '/'

                    values = with_the_folders(args)
                    v = np.asarray(values)
                    l = str(i) + ',' + str(j) + ',' + str(v.mean()) + '\n'
                    f.write(l)
                    print(i, j, v.mean())
