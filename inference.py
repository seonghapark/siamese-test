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

import argparse
from pathlib import Path

from siamese_net import SiameseNetwork
from dataset import SiameseNetworkDataset

def main(args):
    gpu = torch.device(args.device)
    model = SiameseNetwork().cuda(gpu)

    ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Locate the test dataset and load it into the SiameseNetworkDataset
    folder_dataset_test = datasets.ImageFolder(root=args.data_dir)
    transformation = transforms.Compose([transforms.Resize((128,128)),
                                         transforms.ToTensor()])
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                            transform=transformation)
    test_dataloader = DataLoader(siamese_dataset, num_workers=2, batch_size=1, shuffle=True)

    # Grab one image that we are going to test
    dataiter = iter(test_dataloader)
    x0, _, _ = next(dataiter)

    for i in range(5):
        # Iterate over 5 images and test them with the first image (x0)
        _, x1, label2 = next(dataiter)

        # Concatenate the two images together
        concatenated = torch.cat((x0, x1), 0)

        output1, output2 = model(x0.cuda(), x1.cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)
        print(euclidean_distance.item())
        #imshow(torchvision.utils.make_grid(concatenated), f'Dissimilarity: {euclidean_distance.item():.2f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with VICReg", add_help=False)
    parser.add_argument("--data-dir", type=Path, default="/path/to/imagenet", required=True,
                        help='Path to the image net dataset')
    parser.add_argument("--exp-dir", type=Path, default="./exp",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    args = parser.parse_args()
    main(args)
