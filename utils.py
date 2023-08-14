import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
# import RandomErasing
import os
import math
import argparse
import scipy as sp
import scipy.stats
import pickle
import random
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.io import loadmat
from  sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
import torchnet as tnt
from torch.utils.data.dataloader import default_collate
# 自定义测试数据集
import torch.utils.data as data
import torchvision.transforms as transforms


def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):  # 0。9 1。1
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    return alpha * data + beta * noise

def flip_augmentation(data): # arrays tuple 0:(7, 7, 103) 1=(7, 7)
    horizontal = np.random.random() > 0.5 # True
    vertical = np.random.random() > 0.5 # False
    if horizontal:
        data = np.fliplr(data)
    if vertical:
        data = np.flipud(data)
    return data


class hsidataset_target(data.Dataset):
    def __init__(self, data,label):
        self.data=data
        self.label=label
        # self.trans=RandomErasing.RandomErasing()


    def __getitem__(self, index):
        img=self.data[index]
        img1=img
        return img,img1

    def __len__(self):
        return len(self.data)

def twist_loss(p1,p2,alpha=1,beta=1):
    eps=1e-7 #ensure calculate
    #eps=0
    kl_div=((p2*p2.log()).sum(dim=1)-(p2*p1.log()).sum(dim=1)).mean()
    mean_entropy=-(p1*(p1.log()+eps)).sum(dim=1).mean()
    mean_prob=p1.mean(dim=0)
    entropy_mean=-(mean_prob*(mean_prob.log()+eps)).sum()

    return kl_div+alpha*mean_entropy-beta*entropy_mean





