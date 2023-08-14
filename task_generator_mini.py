# code is based on https://github.com/katerakelly/pytorch-maml
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader,Dataset
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import Sampler

def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x

def mini_imagenet_folders():
    train_folder = './datasets/miniImagenet/train'
    test_folder = './datasets/miniImagenet/test'

    metatrain_folders = [os.path.join(train_folder, label) \
                for label in os.listdir(train_folder) \
                if os.path.isdir(os.path.join(train_folder, label)) \
                ]
    metatest_folders = [os.path.join(test_folder, label) \
                for label in os.listdir(test_folder) \
                if os.path.isdir(os.path.join(test_folder, label)) \
                ]

    random.seed(1)
    random.shuffle(metatrain_folders)
    random.shuffle(metatest_folders)

    return metatrain_folders,metatest_folders

class MiniImagenetTask(object):

    def __init__(self, character_folders, num_classes, train_num,test_num):

        self.character_folders = character_folders # list len=64  home/dell/lm/Transferdemo/MyTransferdemo/datas/miniImagenet/train/n04443257'
        self.num_classes = num_classes # 9
        self.train_num = train_num # 1
        self.test_num = test_num # 19

        class_folders = random.sample(self.character_folders,self.num_classes)
        labels = np.array(range(len(class_folders)))  #[0 1 2 3 4 5 6 7 8]
        labels = dict(zip(class_folders, labels))
        samples = dict() #0

        self.train_roots = []
        self.test_roots = []
        for c in class_folders: # '/home/dell/lm/Transferdemo/MyTransferdemo/datas/miniImagenet/train/n03047690'

            temp = [os.path.join(c, x) for x in os.listdir(c)] # list 600
            samples[c] = random.sample(temp, len(temp))  # 600
            random.shuffle(samples[c])

            self.train_roots += samples[c][:train_num]
            self.test_roots += samples[c][train_num:train_num+test_num]

        self.train_labels = [labels[self.get_class(x)] for x in self.train_roots]
        self.test_labels = [labels[self.get_class(x)] for x in self.test_roots]
        # self.train_labels = [labels[x] for x in self.train_roots]
        # self.test_labels = [labels[x] for x in self.test_roots]

    def get_class(self, sample): # '/home/dell/lm/Transferdemo/MyTransferdemo/datas/miniImagenet/train/n03047690/n0304769000000400.jpg'
        a = sample.split('/') #['', 'home', 'dell', 'lm', 'Transferdemo', 'MyTransferdemo', 'datas', 'miniImagenet', 'train', 'n03047690', 'n0304769000000400.jpg']
        b = os.path.join(*sample.split('/')[:-1]) #'home/dell/lm/Transferdemo/MyTransferdemo/datas/miniImagenet/train/n03047690'
        # c = os.path.join('/', b) # '/home/dell/lm/Transferdemo/MyTransferdemo/datas/miniImagenet/train/n03047690'
        # return os.path.join(*sample.split('/')[:-1])
        return b

class FewShotDataset(Dataset):

    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform = transform # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

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
# 自定义数据集
class MiniImagenet(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(MiniImagenet, self).__init__(*args, **kwargs)
        #self.mean_pix = [0.485, 0.456, 0.406]
        #self.std_pix = [0.229, 0.224, 0.225]
        #self.transform_rot = transforms.Compose([
        #    transforms.ToTensor(),
        #    transforms.Normalize(mean=self.mean_pix, std=self.std_pix)
        #])

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)

        #print(image.shape)

        image_rot=image

        image_rot=np.transpose(image_rot, (1, 2, 0))
        #print(image_rot.shape)



        #return rot data label  in fsl dataloader loading rot data
        #split support rot data and query rot data

        #rotated_imgs = [
        #    torch.Tensor(image_rot),
        #    torch.Tensor(rotate_img(image_rot, 90)),
        #    torch.Tensor(rotate_img(image_rot, 180)),
        #    torch.Tensor(rotate_img(image_rot, 270))
        #]

        #rotated_imgs=torch.stack(rotated_imgs, dim=0)

        #print('rotated_imgs',rotated_imgs.shape)

        #rotated_imgs=np.transpose(rotated_imgs,(0,3,1,2))

       # print('new rotated_imgs', rotated_imgs.shape)


        #rotation_labels = torch.LongTensor([0, 1, 2, 3])

        return image, label

# 采样器 重写__iter__()方法
class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_cl, num_inst,shuffle=True):

        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batches = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)] for j in range(self.num_cl)]
        else:
            batches = [[i+j*self.num_inst for i in range(self.num_inst)] for j in range(self.num_cl)]
        batches = [[batches[j][i] for j in range(self.num_cl)] for i in range(self.num_inst)]

        if self.shuffle:
            random.shuffle(batches)
            for sublist in batches:
                   random.shuffle(sublist)
        batches = [item for sublist in batches for item in sublist]
        return iter(batches)

    def __len__(self):
        return 1

class ClassBalancedSamplerOld(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

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


def get_mini_imagenet_data_loader(task, num_per_class=1, split='train',shuffle = False):
    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])

    dataset = MiniImagenet(task,split=split,transform=transforms.Compose([transforms.Resize((33,33)),transforms.ToTensor(),normalize]))
    if split == 'train':
        sampler = ClassBalancedSamplerOld(num_per_class,task.num_classes, task.train_num,shuffle=shuffle)

    else:
        sampler = ClassBalancedSampler(task.num_classes, task.test_num,shuffle=shuffle)

    # dataloader数据加载实现batch，shuffle
    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler) # 使用sampler是，shuffle=false
    return loader
