import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
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
from  sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
#import h5py
#import hdf5storage
import utils
import models
from torch.nn.parameter import Parameter



'''2d DenseNet块'''
# 定义卷积的
def conv_block_2D(in_channel,out_channel):# BN->Relu->conv2d(3x3卷积，padding=1，stride=1，尺寸不变)
    layer = nn.Sequential(
        nn.BatchNorm2d(num_features=in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel,out_channel,kernel_size=3, padding=1, bias=False)
    )
    return layer

class dense_block_2D(nn.Module):

    def __init__(self, in_channel, grow_rate, num_layer):
        '''
        :param in_channel: 输入的通道数
        :param grow_rate:增长率，因为每个卷积层的输出个数都是相等的，有输入和输出的拼接操作，
        每经过一个卷积层，它的输出大小就会增加grow_rate，最后的输出大小为输入大小+层数*grow_rate,是一个增长的过程
        :param num_layer:dense_block中含有卷积层的个数
        '''
        super(dense_block_2D,self).__init__()
        channel = in_channel
        block=[]
        for i in range(num_layer):
            block.append(conv_block_2D(channel, grow_rate))
            channel += grow_rate #Dense ,拼接

        self.net = nn.Sequential(*block)# 定义网络的结构

    def forward(self, x):
        for layer in self.net:
            out = layer(x) # 通过输入计算输出，并存在out里
            x = torch.cat((out,x), dim=1) # 进入下一个卷积单元，将输入拼接到输出里面，这样就实现了一个densenet的基本模块
        return x

'''高光谱空间嵌入特征'''
class SpatialEmbedding(nn.Module):
    '''版本2：简化的双通道高光谱DenseNet网络'''

    def __init__(self):
        super(SpatialEmbedding, self).__init__()
#         self.emb_size = emb_size
#         self.mapping = nn.Conv2d(in_channels=100, out_channels = 3, kernel_size=1, bias=False)
        self.preconv=nn.Conv2d(100,3,1,1,bias=False) # 30 100 50
        self.preconv_bn=nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=0, bias=False)
        self.block = dense_block_2D(32, 12, 3)
        self.conv2 = nn.Conv2d(in_channels=68, out_channels=64, kernel_size=3, padding=0, bias=False)
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, bias=False)

    def forward(self, x):#（1,100,9,9）
#         x = self.mapping(x)#（1,3,9,9）
        x = self.preconv(x)
        x = self.preconv_bn(x)
        x = self.conv1(x) #(1,32,7,7)
        x = self.block(x)#(1,68,7,7)
        x = self.conv2(x) #(1,24,5,5)
        x = F.avg_pool2d(x, kernel_size=5) #(1,24,1,1)
        x = x.view(x.shape[0], -1)  # (10,24)
        return x


'''提取谱嵌入特征'''
class SpectralEmbedding(nn.Module):
    '''版本2：简化的双通道高光谱DenseNet网络'''

    def __init__(self):
        super(SpectralEmbedding, self).__init__()
        #         self.emb_size = emb_size
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=8, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )  # torch.Size([45, 16, 46])
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=4, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )  # torch.Size([45, 16, 21])
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=4, bias=False),
            nn.ReLU(),
            #             nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.5)
        )  # torch.Size([45, 16, 9])
        self.layer4 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, bias=False),
            nn.ReLU(),
            #             nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.5)
        )

    def forward(self, x):  # (45,100)
        x = x.unsqueeze(1)
        x = self.layer1(x)  ##torch.Size([45, 16, 46])
        #         print(x.shape)
        x = self.layer2(x)  # torch.Size([45, 16, 21])
        #         print(x.shape)
        x = self.layer3(x)  ##torch.Size([45, 16, 9])
        #         print(x.shape)
        x = self.layer4(x)
        x = x.squeeze(-1).squeeze(-1)
        return x

# class SpectralEmbedding(nn.Module): # 73.03
#     '''版本2：简化的双通道高光谱DenseNet网络'''
#
#     def __init__(self):
#         super(SpectralEmbedding, self).__init__()
#         #         self.emb_size = emb_size
#         self.layer1 = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=16, kernel_size=16, bias=False),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2)
#         )  # torch.Size([45, 16, 42])
#         self.layer2 = nn.Sequential(
#             nn.Conv1d(in_channels=16, out_channels=16, kernel_size=16, bias=False),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2)
#         )  # torch.Size([45, 16, 13])
#         self.layer3 = nn.Sequential(
#             nn.Conv1d(in_channels=16, out_channels=16, kernel_size=8, bias=False),
#             nn.ReLU(),
#             nn.BatchNorm1d(16),
#             # nn.MaxPool1d(kernel_size=2),
#             nn.Dropout(0.5)
#         )  # torch.Size([45, 16, 9])
#         self.layer4 = nn.Sequential(
#             nn.Conv1d(in_channels=16, out_channels=32, kernel_size=6, bias=False),
#             nn.ReLU(),
#             nn.BatchNorm1d(32),
#             # nn.MaxPool1d(kernel_size=2),
#             nn.Dropout(0.5)
#         )
#
#     def forward(self, x):  # (45,100)
#         x = x.unsqueeze(1)
#         x = self.layer1(x)  ##torch.Size([45, 16, 46])
#         #         print(x.shape)
#         x = self.layer2(x)  # torch.Size([45, 16, 21])
#         #         print(x.shape)
#         x = self.layer3(x)  ##torch.Size([45, 16, 9])
#         #         print(x.shape)
#         x = self.layer4(x)
#         x = x.squeeze(-1).squeeze(-1)
#         return x

class SpectralEmbedding_30(nn.Module):
    '''版本2：简化的双通道高光谱DenseNet网络'''

    def __init__(self):
        super(SpectralEmbedding_30, self).__init__()
        #         self.emb_size = emb_size
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )  # torch.Size([45, 16, 15])
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )  # torch.Size([45, 16, 7])
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            #             nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.5)
        )  # torch.Size([45, 16, 7])
        self.layer4 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            #             nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.5)
        )

    def forward(self, x):  # (45,100)
        x = x.unsqueeze(1)
        x = self.layer1(x)  ##torch.Size([45, 16, 46])
        #         print(x.shape)
        x = self.layer2(x)  # torch.Size([45, 16, 21])
        #         print(x.shape)
        x = self.layer3(x)  ##torch.Size([45, 16, 9])
        #         print(x.shape)
        x = self.layer4(x)
        x = x.squeeze(-1).squeeze(-1)
        return x

class SpectralEmbedding_50(nn.Module):
    '''版本2：简化的双通道高光谱DenseNet网络'''

    def __init__(self):
        super(SpectralEmbedding_50, self).__init__()
        #         self.emb_size = emb_size
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=8, padding=0, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )  # torch.Size([45, 16, 25])
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=4, padding=0, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )  # torch.Size([45, 16, 12])
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=4, padding=0, bias=False),
            nn.ReLU(),
            # nn.BatchNorm1d(16),
            #             nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.5)
        )  # torch.Size([45, 16, 12])
        self.layer4 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=6, bias=False),
            nn.ReLU(),
            # nn.BatchNorm1d(32),
            #             nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.5)
        )

    def forward(self, x):  # (45,100)
        x = x.unsqueeze(1)
        x = self.layer1(x)  ##torch.Size([45, 16, 46])
        #         print(x.shape)
        x = self.layer2(x)  # torch.Size([45, 16, 21])
        #         print(x.shape)
        x = self.layer3(x)  ##torch.Size([45, 16, 9])
        #         print(x.shape)
        x = self.layer4(x)
        x = x.squeeze(-1).squeeze(-1)
        return x

class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        #         x = x.unsqueeze(1)
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x


# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
#         self.enc_nn_spec = SpectralEmbedding()  # 32
#         self.enc_nn_spat = SpatialEmbedding()  # 64
#         self.final_feat_dim = 64+32  # 64+32
#         #         self.bn = nn.BatchNorm1d(self.final_feat_dim)
#         self.classifier = nn.Linear(in_features=self.final_feat_dim, out_features=9, bias=False)
#         self.target_mapping = Mapping(103, 100)
#         self.source_mapping = Mapping(128, 100)
#
#     def forward(self, x, domain='source'):  # x
#
#         if domain == 'target':
#             x = self.target_mapping(x)  # (45, 100,9,9)
#         elif domain == 'source':
#             x = self.source_mapping(x)  # (45, 100,9,9)
#         #         print(x.shape)#torch.Size([45, 100, 9, 9])
#         index = x.shape[2] // 2  # 4
#         data_pixel = x[:, :, index, index]  # (45, 103)
#
#         spec_feature = self.enc_nn_spec(data_pixel)  ## (45, 32)
#         #         print(x.shape)#torch.Size([45, 100, 9, 9])
#         spat_feature = self.enc_nn_spat(x)  # (45, 64)
#         feature = torch.cat([spec_feature, spat_feature], 1)
#         #         feature = self.bn(feature)
#         output = self.classifier(feature)
#         return feature, output

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

class DomainClassifier(nn.Module):
    def __init__(self):# torch.Size([1, 64, 7, 3, 3])
        super(DomainClassifier, self).__init__() # 5 层 线性层
        self.layer = nn.Sequential(
            nn.Linear(1024, 1024), #nn.Linear(320, 512), nn.Linear(FEATURE_DIM*CLASS_NUM, 1024),
#             nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
#             nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
#             nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
#             nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

#             nn.Linear(512, 1),
        )
        self.domain = nn.Linear(1024, 1) # 512
#         self.label = nn.Linear(1024, 9) # 512

    def forward(self, x, iter_num):
        # 512-512-512-512-1
#         x = self.conv(x).view(x.shape[0], -1) # torch.Size([1, 320])
        # x = x.view(x.shape[0], -1) # 没有卷积操作
        coeff = calc_coeff(iter_num, 1.0, 0.0, 10,10000.0)
#         print(coeff)
        x.register_hook(grl_hook(coeff))
        x = self.layer(x)
        domain_y = self.domain(x)
#         label_y = self.label(x)
        return domain_y #,label_y

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]


class learnedweight(nn.Module):
    def __init__(self):
        super(learnedweight, self).__init__()
        self.fsl_weight = Parameter(torch.ones(1), requires_grad=True)  # 1
        self.da_weight = Parameter(torch.ones(1), requires_grad=True)  # 1

    def forward(self, fsl_loss, da_loss):
        #         print(self.fsl_weight)
        #         print(self.da_weight)
        final_loss = self.fsl_weight + torch.exp(-1 * self.fsl_weight) * fsl_loss + self.da_weight + \
                     torch.exp(-1 * self.da_weight) * da_loss
        return final_loss

#############################################################################################################
def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm3d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer


class residual_block(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3x3(in_channel, out_channel)
        self.conv2 = conv3x3x3(out_channel, out_channel)
        self.conv3 = conv3x3x3(out_channel, out_channel)

    def forward(self, x):  # (1,1,100,9,9)
        x1 = F.relu(self.conv1(x), inplace=True)  # (1,8,100,9,9)  (1,16,25,5,5)
        x2 = F.relu(self.conv2(x1), inplace=True)  # (1,8,100,9,9) (1,16,25,5,5)
        x3 = self.conv3(x2)  # (1,8,100,9,9) (1,16,25,5,5)

        out = F.relu(x1 + x3, inplace=True)  # (1,8,100,9,9)  (1,16,25,5,5)
        return out


class D_Res_3d_CNN(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2):
        super(D_Res_3d_CNN, self).__init__()

        self.block1 = residual_block(in_channel, out_channel1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(4, 2, 2), padding=(0, 1, 1), stride=(4, 2, 2))
        self.block2 = residual_block(out_channel1, out_channel2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(4, 2, 2), stride=(4, 2, 2), padding=(2, 1, 1))
        self.conv = nn.Conv3d(in_channels=out_channel2, out_channels=32, kernel_size=3, bias=False)

        # self.final_feat_dim = 160
        #         self.classifier = nn.Sequential(
        #             nn.Linear(self.final_feat_dim, 64),
        #             nn.ReLU(inplace=True),
        #             nn.Dropout(0.5),

        #             nn.Linear(64, 64),
        #             nn.ReLU(inplace=True),
        #             nn.Dropout(0.5),

        #             nn.Linear(64, CLASS_NUM),
        #         )
        # self.classifier = nn.Linear(in_features=self.final_feat_dim, out_features=CLASS_NUM, bias=False)

    def forward(self, x):  # x:(400,100,9,9)
        x = x.unsqueeze(1)  # (400,1,100,9,9)
        x = self.block1(x)  # (1,8,100,9,9)
        x = self.maxpool1(x)  # (1,8,25,5,5)
        x = self.block2(x)  # (1,16,25,5,5)
        x = self.maxpool2(x)  # (1,16,7,3,3)
        x = self.conv(x)  # (1,32,5,1,1)
        x = x.view(x.shape[0], -1)  # (1,160)
        #         x = F.relu(x)
        # y = self.classifier(x)
        return x#, y
