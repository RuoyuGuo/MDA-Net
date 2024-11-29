import os
import random

import torch
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset
from dotmap import DotMap

import json
from utils.utils import np2tensor, bgr2rgb, RandomResizedCropCoord

from torchvision import transforms
import pandas as pd

class REFUGETestDataset(Dataset):
    '''
    every 7 Synthetic low-quality image, has 1 high-quality image
    '''
    
    def __init__(self, cfgs):
        '''
        test on both train and test dataset
        '''
        
        self.lq_path = os.path.join(cfgs.dataset_path, 'test', 'degraded_good', 'de_image')
        self.hq_path = os.path.join(cfgs.dataset_path, 'test', 'resized_good', 'image')
        self.mask_path = os.path.join(cfgs.dataset_path, 'test', 'resized_good', 'mask')
        
        self.namelist = self._init_filename()
        
        self.resize = 0
        self.if_norm = cfgs.if_norm

        if cfgs.resize == 0:
            self.resize = 0
        elif isinstance(cfgs.resize, list):
            self.resize = cfgs.resize
        else:
            self.resize = [cfgs.resize, cfgs.resize]
        
    def _init_filename(self):
        '''
        image index list
        '''
        temp_namelist = os.listdir(self.lq_path)
        temp_namelist = [e.split('.')[0] for e in temp_namelist if '.png' in e]
        
        return temp_namelist
    

    def _load_image(self, idx, v):
        sampled_img = None
        lq_name = self.namelist[idx]

        hq_name = self.namelist[idx]
        hq_name = hq_name.split('_')
        hq_name = hq_name[0]
        
        if v == 'lq':
        #randomly choose degraded type
            sampled_img = cv.imread(os.path.join(self.lq_path, \
                                                 lq_name + '.png'
                                                    ))

            sampled_img = (bgr2rgb(sampled_img), lq_name+'.png')
            
        if v == 'hq':
        #directly return the high-quality image 
            sampled_img = cv.imread(os.path.join(self.hq_path, \
                                                 hq_name + '.png'            
                                                ))
            
            sampled_img = bgr2rgb(sampled_img)
        
        if v == 'mask':
            sampled_img = cv.imread(os.path.join(self.mask_path, \
                                                 hq_name + '.png'            
                                                ), 2)
        return sampled_img


    def _norm(self, sample):
        for key in sample:
            sample[key] = sample[key]/255.0

        return sample
    
    
    def _resize(self, sample):
        for key in sample:
            sample[key] = cv.resize(sample[key], self.resize)  
        
        return sample
        
    def __len__(self):
        return len(self.namelist)

    
    def __getitem__(self, idx):
        #print(f'{idx} is selected')
        lq, name = self._load_image(idx, 'lq')
        mask = self._load_image(idx, 'mask')
        
        sample = {'lq': lq, 'mask': mask}
        
        #resize
        if self.resize != 0:
            sample = self._resize(sample)

        #normalisation
        if self.if_norm is True:
            sample = self._norm(sample)

        #conver to pytorch tensor
        for key, value in sample.items():
            sample[key] = np2tensor(value)
        
        return sample['lq'], sample['mask'], name