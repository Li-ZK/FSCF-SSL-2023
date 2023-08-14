import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from dataset import TensorDataset
from torch.utils.data.sampler import Sampler
import numpy as np
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
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import time
import task_generator_mini as tg
from torch.nn import init
import rot_dataloader
import utils
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-d","--tar_input_dim",type = int, default = 204) # PaviaU=103；salinas=204； IN=200; KSC=176;Botswana = 144
parser.add_argument("-w","--class_num",type = int, default = 16)
parser.add_argument("-s","--shot_num_per_class",type = int, default = 1)
parser.add_argument("-b","--query_num_per_class",type = int, default = 19)  # 19 76 15 75
args = parser.parse_args(args=[])

# Hyper Parameters
TAR_INPUT_DIMENSION = args.tar_input_dim
CLASS_NUM = args.class_num
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
# fine-tuning
TEST_CLASS_NUM = 16
Nlabeled = 5

import torch.utils.data as data

class matcifar(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, imdb, train, d, medicinal):

        self.train = train  # training set or test set
        self.imdb = imdb
        self.d = d
        self.x1 = np.argwhere(self.imdb['set'] == 1)
        self.x2 = np.argwhere(self.imdb['set'] == 3)
        self.x1 = self.x1.flatten()
        self.x2 = self.x2.flatten()
        #        if medicinal==4 and d==2:
        #            self.train_data=self.imdb['data'][self.x1,:]
        #            self.train_labels=self.imdb['Labels'][self.x1]
        #            self.test_data=self.imdb['data'][self.x2,:]
        #            self.test_labels=self.imdb['Labels'][self.x2]

        if medicinal == 1:
            self.train_data = self.imdb['data'][self.x1, :, :, :]
            self.train_labels = self.imdb['Labels'][self.x1]
            self.test_data = self.imdb['data'][self.x2, :, :, :]
            self.test_labels = self.imdb['Labels'][self.x2]

        else:
            self.train_data = self.imdb['data'][:, :, :, self.x1]
            self.train_labels = self.imdb['Labels'][self.x1]
            self.test_data = self.imdb['data'][:, :, :, self.x2]
            self.test_labels = self.imdb['Labels'][self.x2]
            if self.d == 3:
                self.train_data = self.train_data.transpose((3, 2, 0, 1))  ##(17, 17, 200, 10249)
                self.test_data = self.test_data.transpose((3, 2, 0, 1))
            else:
                self.train_data = self.train_data.transpose((3, 0, 2, 1))
                self.test_data = self.test_data.transpose((3, 0, 2, 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:

            img, target = self.train_data[index], self.train_labels[index]
        else:

            img, target = self.test_data[index], self.test_labels[index]

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(0)

import matplotlib.pyplot as plt
def classification_map(map, groundTruth, dpi, savePath):

    fig = plt.figure(frameon=False)
    fig.set_size_inches(groundTruth.shape[1]*2.0/dpi, groundTruth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(savePath, dpi = dpi)

    return 0
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

from operator import truediv
def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

# DATA
def flip(data):
    y_4 = np.zeros_like(data)
    y_1 = y_4
    y_2 = y_4
    first = np.concatenate((y_1, y_2, y_1), axis=1)
    second = np.concatenate((y_4, data, y_4), axis=1)
    third = first
    Data = np.concatenate((first, second, third), axis=0)
    return Data

def load_data(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)

    data_key = image_file.split('/')[-1].split('.')[0]
    label_key = label_file.split('/')[-1].split('.')[0]
    data_all = image_data[data_key]  # dic-> narray , KSC:ndarray(512,217,204)
    GroundTruth = label_data[label_key]

    [nRow, nColumn, nBand] = data_all.shape
    print(data_key, nRow, nColumn, nBand)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  # (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1],data_all.shape[2])

    return Data_Band_Scaler, GroundTruth  # image:(512,217,3),label:(512,217)

def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
    alpha = np.random.uniform(*alpha_range) # Randomly sample from a uniform distribution [low, high)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape) # random Gaussian noise
    return alpha * data + beta * noise

def flip_augmentation(data): # arrays tuple 0:(7, 7, 103) 1=(7, 7)
    horizontal = np.random.random() > 0.5 # True
    vertical = np.random.random() > 0.5 # False
    if horizontal:
        data = np.fliplr(data)  # Flip the array left and right
    if vertical:
        data = np.flipud(data)  # Flip the array upside down
    return data

def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    print(Data_Band_Scaler.shape)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth))
    data_band_scaler = flip(Data_Band_Scaler)  # (1830, 1020, 103)
    groundtruth = flip(GroundTruth)  # (1830, 1020)
    del Data_Band_Scaler
    del GroundTruth

    HalfWidth = 16
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]  # (642, 372)
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth, :]  # (642, 372, 103)

    [Row, Column] = np.nonzero(G)  # (42776,) (42776,)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    print('number of sample', nSample)

    train = {}
    test = {}
    da_train = {}
    m = int(np.max(G))  # 9
    nlabeled = Nlabeled
    print('labeled number:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)  # 40.0
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)  # 40

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)  #
        nb_val = shot_num_per_class  # 4
        train[i] = indices[:nb_val]  #
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):  # 25
            da_train[i] += indices[:nb_val]
        #         da_train[i] = indices[:nb_val] + indices[:nb_val] + indices[:nb_val] + indices[:nb_val] + indices[:nb_val]  #
        test[i] = indices[nb_val:]  # train_test_split

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)
    # return train_indices, test_indices
    print('the number of train_indices:', len(train_indices))  # 520
    print('the number of test_indices:', len(test_indices))  # 9729
    print('the number of train_indices after data argumentation:', len(da_train_indices))  # 520
    #     print(da_train_indices)

    nTrain = len(train_indices)  # 520
    nTest = len(test_indices)  # total_size - nTrain #9729
    da_nTrain = len(da_train_indices)  # 520

    trainX = np.zeros([nTrain,  nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    trainY = np.zeros([nTrain], dtype=np.int64)
    testX = np.zeros([nTest, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    testY = np.zeros([nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices
    RandPerm = np.array(RandPerm)
    for i in range(nTrain):
        trainX[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
                             Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1, :], (2,0,1))
        trainY[i] = G[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    trainY = trainY - 1

    # test:train_loader(45=5*9)
    train_dataset = TensorDataset(data_tensor=trainX, target_tensor=trainY)
    train_loader = DataLoader(train_dataset, batch_size=class_num * shot_num_per_class, shuffle=False, num_workers=0)
    print("train data set is ok")
    del train_dataset

    for i in range(nTest):
        testX[i, :, :, :] = np.transpose(data[ Row[RandPerm[i + nTrain ]] - HalfWidth: Row[RandPerm[i + nTrain]] + HalfWidth + 1, \
                            Column[RandPerm[i + nTrain]] - HalfWidth: Column[RandPerm[ i + nTrain ]] + HalfWidth + 1,:],(2,0,1))
        testY[i] = G[Row[RandPerm[i + nTrain ]], Column[RandPerm[i + nTrain ]]].astype( np.int64)
    print("test data set is ok")
    testY = testY - 1
    test_dataset = TensorDataset(data_tensor=testX, target_tensor=testY)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    print("test  loader and train loader is OK")
    del test_dataset


    # Target domain augmentation data
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],
                                     dtype=np.float32)  # (33, 33, 103, 1800)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):  # radiation_noise，flip_augmentation
        imdb_da_train['data'][:, :, :, iSample] = radiation_noise(
            data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1  # 1-16 -> 0-15
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('ok')

    return train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain, nBand  # , imdb_ori_train
#

test_IP_data = './datasets/salinas/salinas_corrected.mat'
test_IP_label = './datasets/salinas/salinas_gt.mat'

Data_Band_Scaler, GroundTruth = load_data(test_IP_data, test_IP_label)

def get_target_dataset(Data_Band_Scaler, GroundTruth):
    train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain,nBand= get_train_test_loader(Data_Band_Scaler = Data_Band_Scaler, GroundTruth = GroundTruth, \
                        class_num=TEST_CLASS_NUM, shot_num_per_class=Nlabeled)
    train_datas, train_labels = train_loader.__iter__().next()
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape)
    # print('train datas', train_datas[0, 0, :,:])
    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape) #(9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler#,GroundTruth

    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1)) # (9,9,100, 1800)->(1800, 100, 9, 9)
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels'] # (1800,)
    print('target data augmentation label:', target_da_labels)
    target_ssl_hsi_data = list()
    target_ssl_hsi_label = list()

    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):  # labels：ndarray （5200，），data :ndarray (5200,3,64,64)
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)  # path:ndarray (64,64,3)
        target_ssl_hsi_data.append(path)
        target_ssl_hsi_label.append(class_)
    target_da_metatrain_data = target_da_train_set # dice key：value 0：[]
    print(target_da_metatrain_data.keys())
    # print(metatrain_data[0][0].shape)
    return train_loader, test_loader, target_da_metatrain_data, G, RandPerm, Row, Column, nTrain,nBand,target_ssl_hsi_data,target_ssl_hsi_label


class Task(object):

    def __init__(self, data, num_classes, shot_num, query_num):
        self.data = data
        self.num_classes = num_classes
        self.support_num = shot_num
        self.query_num = query_num

        class_folders = sorted(list(data))

        class_list = random.sample(class_folders, self.num_classes)

        labels = np.array(range(len(class_list)))

        labels = dict(zip(class_list, labels))

        samples = dict()

        self.support_datas = []
        self.query_datas = []
        self.support_labels = []
        self.query_labels = []
        for c in class_list:
            temp = self.data[c]  # list
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])

            self.support_datas += samples[c][:shot_num]
            self.query_datas += samples[c][shot_num:shot_num + query_num]

            self.support_labels += [labels[c] for i in range(shot_num)]
            self.query_labels += [labels[c] for i in range(query_num)]

class FewShotDataset(Dataset):
    def __init__(self, task, split='train'):
        self.task = task
        self.split = split
        self.image_datas = self.task.support_datas if self.split == 'train' else self.task.query_datas
        self.labels = self.task.support_labels if self.split == 'train' else self.task.query_labels

    def __len__(self):
        return len(self.image_datas)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

class HBKC_dataset(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(HBKC_dataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image = self.image_datas[idx]
        label = self.labels[idx]
        return image, label


class ClassBalancedSampler(Sampler):
    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1

# dataloader
def get_HBKC_data_loader(task, num_per_class=1, split='train',shuffle = False):
    dataset = HBKC_dataset(task,split=split)

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.support_num, shuffle=shuffle) # support set
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.query_num, shuffle=shuffle) # query set

    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)

    return loader

class FewShotDataset(Dataset):
    def __init__(self, task, split='train'):
        self.task = task
        self.split = split
        self.image_datas = self.task.support_datas if self.split == 'train' else self.task.query_datas
        self.labels = self.task.support_labels if self.split == 'train' else self.task.query_labels

    def __len__(self):
        return len(self.image_datas)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

class HBKC_dataset(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(HBKC_dataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image = self.image_datas[idx]
        label = self.labels[idx]
        return image, label


class ClassBalancedSampler(Sampler):
    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1

# dataloader
def get_HBKC_data_loader(task, num_per_class=1, split='train',shuffle = False):
    dataset = HBKC_dataset(task,split=split)

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.support_num, shuffle=shuffle) # support set
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.query_num, shuffle=shuffle) # query set

    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)

    return loader

# MODEL
class VGG16_SE(nn.Module):
    def __init__(self):
        super(VGG16_SE, self).__init__()
        self.preconv=nn.Conv2d(TAR_INPUT_DIMENSION,3,1,1,bias=False) 
        self.preconv_bn=nn.BatchNorm2d(3)

        self.conv1=nn.Sequential(
             nn.Conv2d(3, 64, 3, padding=1),
             nn.BatchNorm2d(64,eps=1e-05,momentum=0.1, affine=True),
             nn.ReLU(inplace=True)
            )
        self.conv2=nn.Sequential(
             nn.Conv2d(64, 64, 3, padding=1),
             nn.BatchNorm2d(64,eps=1e-05,momentum=0.1, affine=True),
             nn.ReLU(inplace=True)
            )
        self.maxpool2=nn.MaxPool2d(2, stride=2, ceil_mode=False)
        self.conv3=nn.Sequential(
             nn.Conv2d(64, 128, 3, padding=1),
             nn.BatchNorm2d(128,eps=1e-05,momentum=0.1, affine=True),
             nn.ReLU(inplace=True)
            )
        self.conv4=nn.Sequential(
             nn.Conv2d(128, 128, 3, padding=1),
             nn.BatchNorm2d(128,eps=1e-05,momentum=0.1, affine=True),
             nn.ReLU(inplace=True)
            )
        self.maxpool4=nn.MaxPool2d(2, stride=2, ceil_mode=False)

        self.conv5=nn.Sequential(
             nn.Conv2d(128, 256, 3, padding=1),
             nn.BatchNorm2d(256,eps=1e-05,momentum=0.1, affine=True),
             nn.ReLU(inplace=True)
            )
        self.conv6=nn.Sequential(
             nn.Conv2d(256, 256, 3, padding=1),
             nn.BatchNorm2d(256,eps=1e-05,momentum=0.1, affine=True),
             nn.ReLU(inplace=True)
            )
        self.conv7=nn.Sequential(
             nn.Conv2d(256, 256, 3, padding=1),
             nn.BatchNorm2d(256,eps=1e-05,momentum=0.1, affine=True),
             nn.ReLU(inplace=True),
            )
        self.maxpool7=nn.MaxPool2d(2, stride=2, ceil_mode=False)
        self.conv=nn.Sequential(
        nn.Conv2d(256,512,3,1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(2),
        )
        self.fc=nn.Linear(512,100)

    def forward(self, x): 
        x=self.preconv(x)
        x=self.preconv_bn(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.maxpool7(x)
        x=self.conv(x)
        x= x.view(x.size(0), -1)
        x=self.fc(x)
        return x

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1=nn.Sequential(
             nn.Conv2d(3, 64, 3, padding=1),
             nn.BatchNorm2d(64,eps=1e-05,momentum=0.1, affine=True),
             nn.ReLU(inplace=True)
            )
        self.conv2=nn.Sequential(
             nn.Conv2d(64, 64, 3, padding=1),
             nn.BatchNorm2d(64,eps=1e-05,momentum=0.1, affine=True),
             nn.ReLU(inplace=True)
            )
        self.maxpool2=nn.MaxPool2d(2, stride=2, ceil_mode=False)
        self.conv3=nn.Sequential(
             nn.Conv2d(64, 128, 3, padding=1),
             nn.BatchNorm2d(128,eps=1e-05,momentum=0.1, affine=True),
             nn.ReLU(inplace=True)
            )
        self.conv4=nn.Sequential(
             nn.Conv2d(128, 128, 3, padding=1),
             nn.BatchNorm2d(128,eps=1e-05,momentum=0.1, affine=True),
             nn.ReLU(inplace=True)
            )
        self.maxpool4=nn.MaxPool2d(2, stride=2, ceil_mode=False)

        self.conv5=nn.Sequential(
             nn.Conv2d(128, 256, 3, padding=1),
             nn.BatchNorm2d(256,eps=1e-05,momentum=0.1, affine=True),
             nn.ReLU(inplace=True)
            )
        self.conv6=nn.Sequential(
             nn.Conv2d(256, 256, 3, padding=1),
             nn.BatchNorm2d(256,eps=1e-05,momentum=0.1, affine=True),
             nn.ReLU(inplace=True)
            )
        self.conv7=nn.Sequential(
             nn.Conv2d(256, 256, 3, padding=1),
             nn.BatchNorm2d(256,eps=1e-05,momentum=0.1, affine=True),
             nn.ReLU(inplace=True),
            )
        self.maxpool7=nn.MaxPool2d(2, stride=2, ceil_mode=False)
        self.conv=nn.Sequential(
        nn.Conv2d(256,512,3,1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(512, 100)
        self.fc1=nn.Linear(100,4)
        self.relu=nn.ReLU()

    def forward(self, x,type=None): 
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.maxpool7(x)
        x=self.conv(x)
        x= x.view(x.size(0), -1)
        x=self.fc(x)
        if(type=='rot'):
            x=self.relu(x)
            x=self.fc1(x)

        return x

class AttentionLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(AttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1=nn.Linear(channel, channel // reduction, bias=False)
        self.relu=nn.ReLU(inplace=True)
        self.fc2=nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y1=self.fc1(y)
        self.fc1_feature=y1
        y2=self.relu(y1)
        self.relu_feature=y2
        y3=self.fc2(y2)
        self.fc2_feature=y3
        y4=self.sigmoid(y3).view(b, c, 1, 1)
        self.sigmoid_feature=y4
        y5=x * y4.expand_as(x)
        self.feature=y5
        return y5

#1d cnn
def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv1d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm1d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class Spectral_1d(nn.Module):
    def __init__(self,input_channels):
        super(Spectral_1d, self).__init__()
        self.feature_dim = input_channels
        # Convolution Layer 1 kernel_size = (1, 1, 7), stride = (1, 1, 2), output channels = 24
        self.conv1 = nn.Conv1d(1, 24, 7,stride=2,padding=0)
        self.bn1 = nn.BatchNorm1d(24)

        self.activation1 = nn.ReLU()

        # Residual block 1
        self.conv2 = nn.Conv1d(24,24,7,stride=1,padding=3)
        self.bn2 = nn.BatchNorm1d(24)
        self.activation2 = nn.ReLU()

        self.conv3 = nn.Conv1d(24,24,7,stride=1,padding=3)
        self.bn3 = nn.BatchNorm1d(24)
        self.activation3 = nn.ReLU()
        # Finish

        # Convolution Layer 2 kernel_size = (1, 1, (self.feature_dim - 6) // 2), output channels = 128
        self.conv4 = nn.Conv1d(24, 128, (self.feature_dim - 7) // 2 + 1,stride=1,padding=0)
        self.bn5 = nn.BatchNorm1d(128)
        self.activation5 = nn.ReLU()
        self.fc1 = nn.Linear(in_features=128, out_features=100)

    def forward(self,x):
        # Convolution layer 1
        x = self.conv1(x)
        x = self.activation1(self.bn1(x))
        # Residual layer 1
        residual = x
        x = self.conv2(x)
        x = self.activation2(self.bn2(x))
        x = self.conv3(x)
        x = residual + x
        x = self.activation3(self.bn3(x))
        # Convolution layer 2
        x = self.conv4(x)
        x = self.activation5(self.bn5(x))
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x

def convert_vgg(vgg16):
    net = VGG16()
    vgg_items = net.state_dict().items()
    vgg16_items = vgg16.items()
    pretrain_model = {}
    j = 0
    i = 0
    for k,v in list(net.state_dict().items()): # k='preconv.weight';v=torch.Size([3, 200, 1, 1])

        v=list(vgg16_items)[j][1] #torch.Size([64, 3, 3, 3])
        k=list(vgg_items)[i][0] #k=list(vgg_items)[i+6][0] #'preconv_bn.num_batches_tracked' 6 7 / 8 9 10 11 12
        pretrain_model[k]=v
        j = j+1
        if j!=0 and j % 6 == 0:
            i = i + 2
        else:
            i = i + 1

        if j>=42:
            break
    return pretrain_model

def convert_vgg_se(vgg16):
    net = VGG16_SE()
    vgg_items = net.state_dict().items()
    vgg16_items = vgg16.items()
    # for key in vgg16:
    #     print(key, vgg16[key].size())
    pretrain_model = {}
    j = 0
    # i = 0
    for k,v in list(net.state_dict().items()): # k='preconv.weight';v=torch.Size([3, 200, 1, 1])
        v=list(vgg16_items)[j][1] #torch.Size([64, 3, 3, 3])
        k=list(vgg_items)[j+6][0] #k=list(vgg_items)[i+6][0] #'preconv_bn.num_batches_tracked' 6 7 / 8 9 10 11 12
        pretrain_model[k]=v
        j = j+1
        # if j!=0 and j % 6 == 0:
        #     i = i + 2
        # else:
        #     i = i + 1

        if j>=49:
            break
    return pretrain_model

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

rot_dataset_train = rot_dataloader.GenericDataset(
    dataset_name='miniImagenet', 
    split='train',
)

rot_dloader_train=rot_dataloader.DataLoader(
    dataset=rot_dataset_train,
    batch_size=128,
    unsupervised=True,
    epoch_size=None,
    num_workers=0,
    shuffle=True)

#############################################MiniImageNet#######################################################
# train miniImaget
print("Training miniageNet...")

# Step 1: init data folders
print("init data folders")
# init character folders for dataset construction
metatrain_folders, metatest_folders = tg.mini_imagenet_folders()

# model
model=VGG16()
print ('load the weight from vgg')
pretrained_dict = torch.load('./vgg16_bn-6c64b313.pth')
pretrained_dict = convert_vgg(pretrained_dict)

model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
print ('copy the weight sucessfully')

# Loss and Optimizer
crossEntropy = nn.CrossEntropyLoss()
lr=0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
model.cuda()

last_accuracy = 0.0
best_episdoe = 0
train_loss = []
test_acc = []
running_D_loss, running_F_loss = 0.0, 0.0
running_label_loss = 0
running_domain_loss = 0
total_hit, total_num = 0.0, 0.0
test_acc_list = []
train_start = time.time()
for episode in range(1100):  # EPISODE = 500000

    task = tg.MiniImagenetTask(metatrain_folders, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 5， 1，15
    support_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
    query_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)
    supports, support_labels = support_dataloader.__iter__().next()  
    querys, query_labels = query_dataloader.__iter__().next()  
    support_features = model(supports.cuda())  
    query_features = model(querys.cuda())  

    if SHOT_NUM_PER_CLASS > 1:
        support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
    else:
        support_proto = support_features

    logits = euclidean_metric(query_features, support_proto)
    loss_fsl = crossEntropy(logits, query_labels.cuda())

    rot_data, rot_label = rot_dloader_train(0).__iter__().next()
    rot_data, rot_label = rot_data.cuda(), rot_label.cuda()
    output = model(rot_data,type='rot')
    loss_rot = crossEntropy(output, rot_label)

    loss=loss_fsl+loss_rot

    model.zero_grad()

    loss.backward()

    optimizer.step()

    total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
    total_num += querys.shape[0]

    if (episode + 1) % 50 == 0:
        print('episode {:>3d}:  total_loss {:6.6f},fsl_loss {:6.6f},ssl_rot_loss {:6.6f}, acc {:6.6f},'.format(
            episode + 1, loss.item(),loss_fsl.item(),loss_rot.item(), total_hit / total_num))

torch.save(model.state_dict(), "VGG16-Salinas.pth")
del metatrain_folders, metatest_folders



###############################################################HSI#######################################################
# model
class Network(nn.Module):
    def __init__(self, n_bands, n_classes):
        super(Network, self).__init__()
        self.n_bands = n_bands
        self.n_classes = n_classes

        self.spat_model=VGG16_SE()
        print ('load the weight from vgg')
        pretrained_dict = torch.load('./VGG16-Salinas.pth')
        pretrained_dict = convert_vgg_se(pretrained_dict)
        model_dict = self.spat_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.spat_model.load_state_dict(model_dict)
        print ('copy the weight sucessfully')
        print (self.spat_model)

        self.spec_model = Spectral_1d(self.n_bands)
        self.relu=nn.ReLU()
        self.fc = nn.Linear(in_features=100 * 2, out_features=64, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(in_features=64, out_features=self.n_classes, bias=False)


        self.dropout_data_aug = nn.Dropout(0.28) 
        self.twist_batchnorm1d=nn.BatchNorm1d(16,affine=False)
        self.twist_softmax=nn.Softmax(dim=1)




    def forward(self, x=None,x1=None,x2=None,type=None):

        if(type=='ssl'):

            index = x1.shape[2] // 2  
            train_data_spec_x1 = x1[:, :, index, index]  
            train_data_spec_x1 = train_data_spec_x1.unsqueeze(dim=1) 
            spec_feature_x1 = self.spec_model(train_data_spec_x1) 
            spat_feature_x1 = self.spat_model(x1) 
            feature_x1 = torch.cat([spat_feature_x1, spec_feature_x1], 1)
            feature_x1 = self.dropout_data_aug(feature_x1)
            feature_x1 = torch.relu(self.fc(feature_x1))
            feature_x1 = self.classifier(feature_x1)
            output_x1 = self.twist_softmax(self.twist_batchnorm1d(feature_x1))

            index = x2.shape[2] // 2  
            train_data_spec_x2 = x2[:, :, index, index]  
            train_data_spec_x2 = train_data_spec_x2.unsqueeze(dim=1)  
            spec_feature_x2 = self.spec_model(train_data_spec_x2)  
            spat_feature_x2 = self.spat_model(x2)  
            feature_x2 = torch.cat([spat_feature_x2, spec_feature_x2], 1)
            feature_x2 = self.dropout_data_aug(feature_x2)
            feature_x2 = torch.relu(self.fc(feature_x2))
            feature_x2 = self.classifier(feature_x2)
            output_x2 = self.twist_softmax(self.twist_batchnorm1d(feature_x2))
            return output_x1, output_x2

        else:
            index = x.shape[2] // 2  
            train_data_spec = x[:, :, index, index] 
            train_data_spec = train_data_spec.unsqueeze(dim=1)  
            spec_feature = self.spec_model(train_data_spec)  
            spat_feature = self.spat_model(x)  
            feature = torch.cat([spat_feature,spec_feature], 1)
            feature = torch.relu(self.fc(feature))
            feature = self.dropout(feature)
            output = self.classifier(feature)
            return output


nDataSet=10
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, TEST_CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None
seeds = [1330, 1220, 1336, 1337, 1224, 1236, 1226, 1235, 1233, 1229]

for iDataSet in range(nDataSet):
    np.random.seed(seeds[iDataSet])
    global best_predict, train_end, test_end
    train_loader, test_loader, target_da_metatrain_data, G, RandPerm, Row, Column, nTrain,nBand,target_ssl_hsi_data,target_ssl_hsi_label = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth)
    target_spec_dataset_ssl=utils.hsidataset_target(target_ssl_hsi_data,label=None)

    target_ssl_spec_dataloader=torch.utils.data.DataLoader(target_spec_dataset_ssl,batch_size=128,shuffle=True)

    model = Network(TAR_INPUT_DIMENSION, CLASS_NUM)
    # Loss and Optimizer
    crossEntropy = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.cuda()
    print("HSI Training...")
    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    best_predict_all = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit, total_num = 0.0, 0.0
    test_acc_list = []
    for episode in range(1000):

        if (episode+1)%10==0:
            print('--------------------------EPISODES:',episode+1)

        task = Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 5， 1，15
        support_dataloader = get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
        query_dataloader = get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)
        supports, support_labels = support_dataloader.__iter__().next()  
        querys, query_labels = query_dataloader.__iter__().next() 
        support_features = model(x=supports.cuda())
        query_features = model(x=querys.cuda())  

        if SHOT_NUM_PER_CLASS > 1:
            support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
        else:
            support_proto = support_features

        logits = euclidean_metric(query_features, support_proto)
        loss_fsl = crossEntropy(logits, query_labels.cuda())


        x1,x2=target_ssl_spec_dataloader.__iter__().next()
        p1, p2 = model(x1=x1.type(torch.FloatTensor).cuda(), x2=x2.type(torch.FloatTensor).cuda(), type='ssl')
        loss_spec_ssl = (utils.twist_loss(p1, p2) + utils.twist_loss(p2, p1)) * 0.5

        loss = loss_fsl+loss_spec_ssl

        model.zero_grad()
        loss.backward()
        optimizer.step()

        total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
        total_num += querys.shape[0]

        if (episode + 1) % 20 == 0:
            print('episode {:>3d}:  total loss: {:6.4f},fsl loss: {:6.7f},spec ssl loss: {:6.4f}, acc {:6.4f}'.format(episode + 1, loss.item(),loss_fsl.item(),loss_spec_ssl.item(), total_hit/ total_num))

        if (episode + 1) % 20 == 0 or episode == 0:
            train_loss.append(loss.item())
            # test
            train_end = time.time()
            print("Testing ...")
            model.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)
            best_predict = np.array([], dtype=np.int64)
            train_datas, train_labels = train_loader.__iter__().next()
            train_features = model(x=Variable(train_datas).cuda())  
            max_value = train_features.max()  
            min_value = train_features.min()  
            print(max_value.item())
            print(min_value.item())
            train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

            KNN_classifier = KNeighborsClassifier(n_neighbors=1)
            KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)  
            for test_datas, test_labels in test_loader:
                batch_size = test_labels.shape[0]
                test_features = model(x=Variable(test_datas).cuda())  
                test_features = (test_features - min_value) * 1.0 / (max_value - min_value)
                predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())  
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]
                total_rewards += np.sum(rewards)
                counter += batch_size
                predict = np.append(predict, predict_labels)
                labels = np.append(labels, test_labels.numpy())
                accuracy = total_rewards / 1.0 / counter  #
                accuracies.append(accuracy)

            test_accuracy, h = mean_confidence_interval(accuracies)
            print("[{}:{}],test accuracy={}:".format(total_rewards, counter, test_accuracy))
            test_acc.append(test_accuracy)

            confusion_matrix = metrics.confusion_matrix(labels, predict)
            each_acc, average_acc = AA_andEachClassAccuracy(confusion_matrix)
            kappa = metrics.cohen_kappa_score(labels, predict)
            print('OA= %.5f AA= %.5f k= %.5f' % (accuracy, average_acc, kappa))
            print('each_acc', each_acc)
            test_end = time.time()
            model.train()
            if test_accuracy > last_accuracy:
                last_accuracy = test_accuracy
                best_episdoe = episode

                acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
                OA = acc
                C = metrics.confusion_matrix(labels, predict)
                A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=float)

                k[iDataSet] = metrics.cohen_kappa_score(labels, predict)

            print('best episode:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))

    if test_accuracy > best_acc_all:
        best_predict_all = predict
        best_G, best_RandPerm, best_Row, best_Column, best_nTrain = G, RandPerm, Row, Column, nTrain
    print('***********************************************************************************')
    del train_loader,test_loader,target_ssl_spec_dataloader,query_dataloader,support_dataloader,target_spec_dataset_ssl,target_ssl_hsi_data,target_ssl_hsi_label,task,target_da_metatrain_data


AA = np.mean(A, 1)

AAMean = np.mean(AA, 0)
AAStd = np.std(AA)

AMean = np.mean(A, 0)
AStd = np.std(A, 0)

OAMean = np.mean(acc)
OAStd = np.std(acc)

kMean = np.mean(k)
kStd = np.std(k)
print("train time per DataSet(s): " + "{:.5f}".format(train_end - train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end - train_end))
print("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd))
print("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print("average kappa: " + "{:.4f}".format(100 * kMean) + " +- " + "{:.4f}".format(100 * kStd))
print("accuracy for each class: ")
for i in range(CLASS_NUM):
    print("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))

best_iDataset = 0
for i in range(len(acc)):
    print('{}:{}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('best acc all={}'.format(acc[best_iDataset]))

#################classification map################################
print('classification map !!!!!!!!!!!!!!!!!!!!!')

for i in range(len(best_predict_all)):  
    best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = \
    best_predict_all[i] + 1

hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
for i in range(best_G.shape[0]):
    for j in range(best_G.shape[1]):
        if best_G[i][j] == 0:
            hsi_pic[i, j, :] = [0, 0, 0]
        if best_G[i][j] == 1:
            hsi_pic[i, j, :] = [0, 0, 1]
        if best_G[i][j] == 2:
            hsi_pic[i, j, :] = [0, 1, 0]
        if best_G[i][j] == 3:
            hsi_pic[i, j, :] = [0, 1, 1]
        if best_G[i][j] == 4:
            hsi_pic[i, j, :] = [1, 0, 0]
        if best_G[i][j] == 5:
            hsi_pic[i, j, :] = [1, 0, 1]
        if best_G[i][j] == 6:
            hsi_pic[i, j, :] = [1, 1, 0]
        if best_G[i][j] == 7:
            hsi_pic[i, j, :] = [0.5, 0.5, 1]
        if best_G[i][j] == 8:
            hsi_pic[i, j, :] = [0.65, 0.35, 1]
        if best_G[i][j] == 9:
            hsi_pic[i, j, :] = [0.75, 0.5, 0.75]
        if best_G[i][j] == 10:
            hsi_pic[i, j, :] = [0.75, 1, 0.5]
        if best_G[i][j] == 11:
            hsi_pic[i, j, :] = [0.5, 1, 0.65]
        if best_G[i][j] == 12:
            hsi_pic[i, j, :] = [0.65, 0.65, 0]
        if best_G[i][j] == 13:
            hsi_pic[i, j, :] = [0.75, 1, 0.65]
        if best_G[i][j] == 14:
            hsi_pic[i, j, :] = [0, 0, 0.5]
        if best_G[i][j] == 15:
            hsi_pic[i, j, :] = [0, 1, 0.75]
        if best_G[i][j] == 16:
            hsi_pic[i, j, :] = [0.5, 0.75, 1]

classification_map(hsi_pic[16:-16, 16:-16, :], best_G[16:-16, 16:-16], 24,"classificationMap/Salinas_{}shot.png".format(Nlabeled))
print("train_end_time = ", train_end)