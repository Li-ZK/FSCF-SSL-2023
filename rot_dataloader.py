from __future__ import print_function
import torch
import torch.utils.data as data
import torchvision
import torchnet as tnt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import random
from torch.utils.data.dataloader import default_collate
from PIL import Image
import os
import errno
import numpy as np
import sys
import csv
from PIL import ImageFilter
from pdb import set_trace as breakpoint

# Set the paths of the datasets here.
_CIFAR_DATASET_DIR = './datasets/CIFAR'
_IMAGENET_DATASET_DIR = '/home/dell/桌面/SSL-HFSL-1.19-ROT/HFSL-main/datasets/miniImagenet'
_PLACES205_DATASET_DIR = './datasets/Places205'


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds



class GenericDataset(data.Dataset):
    def __init__(self, dataset_name, split,
                 num_imgs_per_cat=None):
        self.split = split.lower()
        self.dataset_name =  dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
        self.num_imgs_per_cat = num_imgs_per_cat

        if self.dataset_name=='miniimagenet':
            assert(self.split=='train' or self.split=='val')
            self.mean_pix = [0.485, 0.456, 0.406]
            self.std_pix = [0.229, 0.224, 0.225]
            transforms_list = [
                transforms.Resize(33),
                transforms.RandomResizedCrop(33),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x)]
            self.transform = transforms.Compose(transforms_list)
            split_data_dir = _IMAGENET_DATASET_DIR + '/' + self.split
            self.data = datasets.ImageFolder(split_data_dir, self.transform)

    def __getitem__(self, index):
        img, label = self.data[index]
        return img, int(label)

    def __len__(self):
        return len(self.data)

class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def rotate_img(img, rot):
    if rot == 0: # 0 degrees rotation
        img1=img.copy()
        return img1
    elif rot == 90: # 90 degrees rotation
        #img1=np.flipud(img).copy()
        img1=np.flipud(np.transpose(img, (1,0,2))).copy()
        return img1
    elif rot == 180: # 90 degrees rotation
        img1=np.fliplr(np.flipud(img)).copy()
        return img1
    elif rot == 270: # 270 degrees rotation / or -90
        #img1=np.flipud(img).copy()
        img1=np.transpose(np.flipud(img), (1,0,2)).copy()
        return img1
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


class DataLoader(object):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 unsupervised=True,
                 epoch_size=None,
                 num_workers=0,
                 shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.batch_size = batch_size
        self.unsupervised = unsupervised
        self.num_workers = num_workers

        mean_pix  = self.dataset.mean_pix
        std_pix   = self.dataset.std_pix
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_pix, std=std_pix)
        ])
        self.inv_transform = transforms.Compose([
            Denormalize(mean_pix, std_pix),
            lambda x: x.numpy() * 255.0,
            lambda x: x.transpose(1,2,0).astype(np.uint8),
        ])

    def get_iterator(self, epoch=0):
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)
        if self.unsupervised:
            # if in unsupervised mode define a loader function that given the
            # index of an image it returns the 4 rotated copies of the image
            # plus the label of the rotation, i.e., 0 for 0 degrees rotation,
            # 1 for 90 degrees, 2 for 180 degrees, and 3 for 270 degrees.
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img0, _ = self.dataset[idx]

                #print(img0.shape)

                rotated_imgs = [
                    self.transform(img0),
                    self.transform(rotate_img(img0,  90)),
                    self.transform(rotate_img(img0, 180)),
                    self.transform(rotate_img(img0, 270))
                ]
                rotation_labels = torch.LongTensor([0, 1, 2, 3])

                rotated_imgs=torch.stack(rotated_imgs, dim=0)

                #print('old rotated_img',rotated_imgs.shape)
                return rotated_imgs, rotation_labels
            def _collate_fun(batch):
                batch = default_collate(batch)
                assert(len(batch)==2)
                batch_size, rotations, channels, height, width = batch[0].size()
                batch[0] = batch[0].view([batch_size*rotations, channels, height, width])
                batch[1] = batch[1].view([batch_size*rotations])
                return batch
        else: # supervised mode
            # if in supervised mode define a loader function that given the
            # index of an image it returns the image and its categorical label
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img, categorical_label = self.dataset[idx]
                img = self.transform(img)
                return img, categorical_label
            _collate_fun = default_collate

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
            load=_load_function)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
            collate_fn=_collate_fun, num_workers=self.num_workers,
            shuffle=self.shuffle)
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size / self.batch_size

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    dataset = GenericDataset('miniImagenet','train')
    dataloader = DataLoader(dataset, batch_size=8, unsupervised=True)

    print(len(dataloader.get_iterator(8)))

    b=dataloader(0).__iter__().next()

    data, label = b
    print(data)
    print(label)




    inv_transform = dataloader.inv_transform
    for i in range(data.size(0)):
        plt.subplot(data.size(0)/4,4,i+1)
        fig=plt.imshow(inv_transform(data[i]))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.show()

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

