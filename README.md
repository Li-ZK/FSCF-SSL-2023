## Few-shot Hyperspectral Image Classification with Self-supervised Learning
This is a code demo for the paper "Few-shot Hyperspectral Image Classification with Self-supervised Learning"
Zhaokui Li, Hui Guo, Yushi Chen, Cuiwei Liu, Qian Du, Zhuoqun Fang, and Yan Wang, Few-shot Hyperspectral Image Classification with Self-supervised Learning, IEEE Transactions on Geoscience and Remote Sensing.

## Requirements
- CUDA = 11.1
- Python = 3.9 
- Pytorch = 1.8.0
- sklearn = 1.0.1
- numpy = 1.21.2

## Datasets
You can download the hyperspectral datasets in mat format at: http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes, and move the files to `./datasets` folder.
You can also download the hyperspectral datasets from the following link.
Link: https://pan.baidu.com/s/1k8by5CiyabXRJdD_MVOL1w 
Extract code: wjvv

The mini-ImageNet data sets can be downloaded from the following link:
Link: https://pan.baidu.com/s/1Mn1en9EhfFvE-i62YnbwhQ
Extract code: 54DO

An example dataset folder has the following structure:
```
datasets
├── IP
│   ├── indian_pines_corrected.mat
│   ├── indian_pines_gt.mat
└── paviaU
│   ├── paviaU.mat
│   ├── paviaU_gt.mat
└── HC
│   ├── WHU_Hi_HanChuan.mat
│   ├── WHU_Hi_HanChuan(15%)_gt.mat
└── Salinas
│   ├── Salinas_corrected.mat
│   ├── salinas_gt.mat
└──miniImagenet
│   ├── 
│   ├── 

```
## Usage:
Take FSCF-SSL method : 
1. Download the required data set and move to folder`./datasets`.
2. To run the file, you need to download the VGG pre-training weights file (vgg16_bn-6c64b313.pth).
   The VGG pre-training weight file can be downloadfrom the following link:
   Link: https://pan.baidu.com/s/1af--So40MKjhWdFuIcVyKg 
   Extract code：0tdu
3. Taking 5 labeled samples per class as an example, run `FSCF-SSL.py`
