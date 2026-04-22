import torch
#We choose 1500 samples.
#wrn_28_2640
import argparse
import os

import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegressionCV
import numpy as np
import time
from util.metrics import compute_traditional_ood, compute_in
from util.args_loader import get_args
from util.data_loader import get_loader_in, get_loader_out
from util.model_loader import get_model
from score import get_score
import torch
import torch
import torchvision as tv
import time
import numpy as np
import models
import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torch.autograd import Variable
import torch.nn as nn
import math
from models.wrn import WideResNet
import os, sys, time
import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

import copy
from torchvision import transforms, utils
#from cifar_resnet import *

from PIL import Image


class ImageListDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, root_path, imglist, transform=None, target_transform=None):
        self.root_path = root_path
        self.transform = transform
        self.target_transform = target_transform
        with open(imglist) as f:
            self._indices = f.readlines()
    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index):
        img_path, label = self._indices[index].strip().split()
        img_path = os.path.join(self.root_path, img_path)
        img = Image.open(img_path).convert('RGB')
        label = int(label)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def make_id_ood_CIFAR(args):
    """Returns train and validation datasets."""
    # crop = 480
    # crop = 32

    imagesize = 32
    val_tx = tv.transforms.Compose([
        tv.transforms.Resize((imagesize, imagesize)),
        tv.transforms.CenterCrop(imagesize),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    in_set = tv.datasets.CIFAR100("./ID_OOD_dataset/",
                                   train=False,
                                   transform=val_tx,
                                   download=True)
    in_loader = torch.utils.data.DataLoader(in_set, batch_size=25, shuffle=False, num_workers=4)


    return in_set, 0, in_loader, 0


def load_model(num_classes,args):
    print("=> creating model '{}'".format(args.arch))
    net = models.__dict__[args.arch](args.depth, args.wide, num_classes)
    print("=> network :\n {}".format(net))
    # net = torch.nn.DataParallel(net.cuda(), device_ids=list(range(args.ngpu)))
    net = net.cuda()
    trainable_params = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([p.numel() for p in trainable_params])
    print("Number of parameters: {}".format(params))
    return net


def main(args):


    torch.backends.cudnn.benchmark = True

    if args.dataset == 'CIFAR100':
        # model = resnet18_cifar(num_classes=10)
        # model.load_state_dict(torch.load("./checkpoints/resnet18_cifar10.pth")['state_dict'])
        from models.wrn import WideResNet
        model = WideResNet(depth=28, num_classes=100, widen_factor=2, dropRate=0.3)
        numc=100
        #model = load_model(numc,args)
        if args.optimizer == 'sgd': optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, weight_decay=args.decay, momentum=args.momentum, nesterov=True)




        path='checkpoint/cifar100_wrn_standard_epoch_99.pt'
        if os.path.isfile(path):
            print("=> loading checkpoint '{}'".format(path))

            checkpoint = torch.load(path)

            #recorder = checkpoint['recorder']
            #recorder.refresh(args.epochs)
            #args.start_epoch = checkpoint['epoch']



            '''weights_dict = {}
            for k, v in checkpoint['state_dict'].items():
                new_k = k.replace('module.', '') if 'module' in k else k
                weights_dict[new_k] = v

            model.load_state_dict(weights_dict)'''

            #model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            #best_acc = recorder.max_accuracy(False)

            #print("=> loaded checkpoint '{}' accuracy={} (epoch {})" .format(path, best_acc, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(path))

        #    load WRN-50-2:
        #model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
        model = model.cuda()
        # for name,param in model.named_parameters():
        #     print(name)
        in_set, out_set, in_loader, out_loader = make_id_ood_CIFAR(args)
        numc=10

        list_id = torch.zeros((128,1500)) #1500

        for batch_idx, (inputs, true_class) in enumerate(in_loader): #batch_size1
            model.eval()

            #Choose 1500 samples.
            if batch_idx==1500:
                break
            inputs = inputs.cuda()

            #Get the features (640 channels).
            features = model.forward_features(inputs) #【1,640】
            #print("Feature shape:", features.shape)
            #
            current_outputs = features.reshape(features.shape[0],-1).detach() #【1,640】

            for j in range(current_outputs.shape[0]):
                for k in range(128):
                    list_id[k][batch_idx] = current_outputs[j][k] #j,k

        features_mean=torch.zeros(128)
        features_std=torch.zeros(128)
        for i in range(128): #
            features_mean[i]=list_id[i].mean()
            features_std[i]=list_id[i].std()

        torch.save(features_mean,"feat-wrn/wrn_28_100_features_mean.pt")
        torch.save(features_std,"feat-wrn/wrn_28_100_features_std.pt")







if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_datadir",default="/data/gwj/OOD/LAPS-main/datasets/id_data/ILSVRC-2012/val", help="Path to the in-distribution data folder.")
    parser.add_argument("--out_datadir",default="/data/gwj/OOD/LAPS-main/datasets/ood_data/svhn" ,help="Path to the out-of-distribution data folder.")

    parser.add_argument('--score', choices=['MSP', 'ODIN', 'Energy', 'GradNorm', 'MaxLogit'], default='Energy')

    parser.add_argument('--dataset', choices=['CIFAR','CIFAR100',  'ImageNet'], default='CIFAR100')
    parser.add_argument('--bats', default=0, type=int, help='Using BATS to boost the performance or not.')
    # arguments for ODIN
    parser.add_argument('--temperature_odin', default=1000, type=int,
                        help='temperature scaling for odin')
    parser.add_argument('--epsilon_odin', default=0.005, type=float,
                        help='perturbation magnitude for odin')

    # arguments for Energy
    parser.add_argument('--temperature_energy', default=1, type=int,
                        help='temperature scaling for energy')

    # arguments for GradNorm
    parser.add_argument('--temperature_gradnorm', default=1, type=int,
                        help='temperature scaling for GradNorm')
    parser.add_argument('--arch', type=str, default='wide resnet', help='Model architecture: (default: wide resnet)')
    parser.add_argument('--depth', type=int, default=28, help='Depth of the model')
    parser.add_argument('--wide', type=int, default=10, help='Widening factor for Wide ResNets')

    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimization method to train the neural network.')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for the optimizer.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')

    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')

    parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')

    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--start_epoch', type=int, default=0, help='Manual epoch number (useful on restarts)')

    main(parser.parse_args())