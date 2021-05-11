from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import AAnet_transforms
from torch.utils.data import Dataset
from utils import utils
from utils.file_io import read_img, read_disp
from utils.kitti_util import get_depth_map , read_label
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class e2e_dataset(Dataset):
    def __init__(self ,data_dir,mode):
        self.mode = mode
        self.samples_paths = []
        sample = dict()
        
        if mode == 'train':

            #transform_list = [AAnet_transforms.RandomCrop(args.img_height, args.img_width),
            #                AAnet_transforms.RandomColor(),
            #                AAnet_transforms.RandomVerticalFlip(),
            #                AAnet_transforms.ToTensor(),
            #                AAnet_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            #                ]
            transform_list = [AAnet_transforms.ToTensor(),
                              AAnet_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
            self.transforms = AAnet_transforms.Compose(train_transform_list)
            filenames = open("dataset/train_filenames.txt").read().split('\n')
            leftImagesDir = 'dataset/training/image_2/'
            rigthImagesDir =  'dataset/training/image_3/'
            VelodyneDir = 'dataset/training/velodyne/'
            CalibDir = 'dataset/training/calib/'
            LabelDir = 'dataset/training/label_2/'
        else:
            filenames = open("dataset/val_filenames.txt").read().split('\n')
            leftImagesDir = 'dataset/testing/image_2/'
            rigthImagesDir =  'dataset/testing/image_3/'
            VelodyneDir = 'dataset/testing/velodyne/'
            CalibDir = 'dataset/testing/calib/'
            LabelDir = 'dataset/testing/label_2'
            #transform_list = [AAnet_transforms.RandomCrop(args.val_img_height, args.val_img_width,                                     validate=True),
            #              AAnet_transforms.ToTensor(),
            #              AAnet_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            #             ]
            transform_list =[ AAnet_transforms.ToTensor(),
                              AAnet_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                            ]
        self.transforms = AAnet_transforms.Compose(transform_list)
    
        for name in filenames:
                sample = dict()
                sample['left_name'] = name
                sample['left']  = leftImagesDir  + name + '.png'
                sample['right']  = rigthImagesDir  + name + '.png'
                sample['velo']  =  VelodyneDir + name + '.bin'
                sample['calib'] = CalibDir + name + '.txt'
                sample['label'] = LabelDir + name + '.txt'
                self.samples_paths.append(sample)
        
    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples_paths[index]
        sample['left_name'] = sample_path['left_name']

        sample['left'] = read_img(sample_path['left'])  # [H, W, 3]
        sample['right'] = read_img(sample_path['right'])

        sample['disp'] = get_depth_map(index , self.samples_paths)
        
        #label
        sample['label'] = read_label(index , self.samples_paths)
        
        sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return len(self.samples)