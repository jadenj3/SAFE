import numpy as np
import os
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels

# ImageNet-A is the version defined at https://github.com/zhoudw-zdw/RevisitingCIL from here: 
#   @article{zhou2023revisiting,
#        author = {Zhou, Da-Wei and Ye, Han-Jia and Zhan, De-Chuan and Liu, Ziwei},
#        title = {Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need},
#        journal = {arXiv preprint arXiv:2303.07338},
#        year = {2023}
#    }

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

def build_transform(is_train, args,isCifar=False):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        if isCifar:
            size = input_size
        else:
            size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    # return transforms.Compose(t)
    return t

class iCIFAR224(iData):
    use_path = False

    train_trsf=build_transform(True, None,True)
    test_trsf=build_transform(False, None,True)
    common_trsf = [
        # transforms.ToTensor(),
    ]

    class_order = np.arange(100).tolist()

    def __init__(self,use_input_norm):
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def download_data(self):
        do_download=True
        if os.path.isfile('/data/cifar-100-python/train'):
            do_download=False
        train_dataset = datasets.cifar.CIFAR100("/data/", train=True, download=do_download)
        test_dataset = datasets.cifar.CIFAR100("/data/", train=False, download=False)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class iImageNetR(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def __init__(self,use_input_norm):
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def download_data(self):
        # as per Zhou et al (2023), download from https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EU4jyLL29CtBsZkB6y-JSbgBzWF5YHhBAUz1Qw8qM2954A?e=hlWpNW
        train_dir = "/data/imagenet_r/train"
        test_dir = "/data/imagenet_r/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetA(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def __init__(self,use_input_norm):
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def download_data(self):
        # as per Zhou et al (2023), download from  https://drive.google.com/file/d/19l52ua_vvTtttgVRziCZJjal0TPE9f2p/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/ERYi36eg9b1KkfEplgFTW3gBg1otwWwkQPSml0igWBC46A?e=NiTUkL
        train_dir = "/data/imagenet-a/train/"
        test_dir = "/data/imagenet-a/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
