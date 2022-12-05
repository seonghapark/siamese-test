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

from siamresnet import SiamResNet
#from siamresnet import SiamResNet
from siamese_net import SiameseNetwork
from inference_dataset import BasicDataset

from pathlib import Path
import argparse
import os
import glob
import sys
import json

import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Main():
    def __init__(self, args):
        self.gpu = torch.device(args.device)
        #self.model = SiameseNetwork().cuda(self.gpu)
        self.model = SiamResNet(args).cuda(self.gpu)

        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        self.model.load_state_dict(ckpt["model"])





        self.model_weights = []
        self.conv_layers = []
        model_children = list(self.model.backbone.children())
        print(model_children)

        counter = 0
        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv2d:
                counter += 1
                self.model_weights.append(model_children[i].weight)
                self.conv_layers.append(model_children[i])
            elif type(model_children[i]) == nn.Sequential:
                for j in range(len(model_children[i])):
                    for child in model_children[i][j].children():
                        if type(child) == nn.Conv2d:
                            counter += 1
                            self.model_weights.append(child.weight)
                            self.conv_layers.append(child)
        print(f'Total convolutional layers: {counter}')
        print(self.conv_layers)








        self.model.eval()

    def run(self, args, train_loader, val_loader):
        for step, (x, y) in enumerate(zip(train_loader, val_loader), len(val_loader)):
            x = x.cuda(self.gpu, non_blocking=True)
            y = y.cuda(self.gpu, non_blocking=True)

            output1, output2 = self.model(x, y)
            euclidean_distance = F.pairwise_distance(output1, output2)

        return euclidean_distance.item()


    def run(self, args, image):
        x = image.cuda(self.gpu, non_blocking=True)




        outputs = []
        names = []
        for layer in self.conv_layers[0:]:
            x = layer(x)
            outputs.append(x)
            names.append(str(layer))
        print(len(outputs))
        for feature_map in outputs:
            print(feature_map.shape)


        processed = []
        for feature_map in outputs:
            feature_map = feature_map.squeeze(0)
            gray_scale = torch.sum(feature_map, 0)
            print(gray_scale.shape)
            gray_scale = gray_scale / feature_map.shape[0]
            processed.append(gray_scale.data.cpu().numpy())
        for fm in processed:
            print(fm.shape)

        fig = plt.figure(figsize=(100,100))
        for i in range(len(processed)):
            print(processed[i])
            a = fig.add_subplot(7,7,i+1)
            imgplot = plt.imshow(processed[i])
            a.axis('off')
            a.set_title(names[i].split('(')[0], fontsize=30)
        plt.savefig('feature_maps.jpg', bbox_inches='tight')






#         output1 = self.model.forward_once(x)
#         print(output1[0], output1.shape)
        exit(0)

        return output1



def with_the_folder(args):
    name = args.data_dir1 + '*'
    files1 = sorted(glob.glob(name))

    transformation = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize((128,128)),
                                         transforms.ToTensor()])

    pred_array = []
    for i in files1:
        print(i)
        image = cv2.imread(i)
        image = transformation(image).unsqueeze(0)
        image = image.to(device='cuda')
        pred = main.run(args, image)
        pred_array.append(pred)

    return pred_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with VICReg", add_help=False)
    parser.add_argument("--arch", type=str, default="resnet50",
                        help='Architecture of the backbone encoder network')
    parser.add_argument("--data-dir", type=str, default="/path/to/imagenet", required=True,
                        help='Path to the image net dataset')
    parser.add_argument("--exp-dir", type=Path, default="./exp",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    args = parser.parse_args()
    main = Main(args)










    folders = sorted(os.listdir(args.data_dir))

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
    print(" ".join(sys.argv))
    print(" ".join(sys.argv), file=stats_file)

    features = {}
    for i in folders:
        args.data_dir1 = args.data_dir + i + '/'
        values = with_the_folder(args)
        v = np.asarray(values)
        features[int(i)] = v
        print(features)
    print(features)



    '''
    #### only same folders
    outputfile = open('dissimilarity_same.txt', 'a', buffering=1)
    for i in folders:
        args.data_dir1 = args.data_dir2 = args.data_dir + i + '/'
        values = with_the_folders(args)
        v = np.asarray(values)
    '''

    '''
    #### only different folders
    outputfile = open('dissimilarity_diff.txt', 'a', buffering=1)
    for i in folders:
        for j in folders:
            if i == j:
                pass
            elif int(i) > lower_limit and int(i) <= upper_limit and int(j) > lower_limit and int(j) <= upper_limit:
                args.data_dir1 = args.data_dir + i + '/'
                args.data_dir2 = args.data_dir + j + '/'

                values = with_the_folders(args)
                v = np.asarray(values)

                l = str(i) + ',' + str(j) + ',' + str(v.mean())

                print(json.dumps(l))
                print(json.dumps(l), file=outputfile)
    '''