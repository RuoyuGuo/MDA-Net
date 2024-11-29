import os
import random

import torch
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset
from dotmap import DotMap

import json
from utils.utils import np2tensor, bgr2rgb, RandomResizedCropCoord, rgb2lab, rgb2hsv, BaseDataset

from torchvision import transforms
import pandas as pd

class EyeQTrainDataset(BaseDataset):
    '''
    every 7 low-quality image, has 1 high-quality image
    '''
    
    def __init__(self, cfgs):
        self.lq_path = os.path.join(cfgs.dataset_path, 'train', 'degraded_good', 'de_image')
        self.hq_path = os.path.join(cfgs.dataset_path, 'train', 'resized_good', 'image')
        self.namelist = self._init_filename()
        
        self.if_aug = cfgs.if_aug
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
        temp_namelist = os.listdir(self.hq_path)
        temp_namelist = [e.split('.')[0] for e in temp_namelist if '.png' in e]
        
        return temp_namelist
    

    def _load_image(self, idx, v):
        sampled_img = None
        if v == 'lq':
        #randomly choose degraded type
            r = random.randint(0, 15)
                
            #print(f'type {self.de_type[r]} is selected')
            sampled_img = cv.imread(os.path.join(self.lq_path, \
                                                 self.namelist[idx] + '_'
                                                                    + str(r)
                                                                    + '.png'
                                                ))
            
            sampled_img = bgr2rgb(sampled_img)
            
        if v == 'hq':
        #directly return the high-quality image 
            sampled_img = cv.imread(os.path.join(self.hq_path, \
                                                 self.namelist[idx] + '.png'            
                                                ))
            
            sampled_img = bgr2rgb(sampled_img)
            
        return sampled_img
    
    def __len__(self):
        return len(self.namelist)
    
    def __getitem__(self, idx):
        #print(f'{idx} is selected')
        lq = self._load_image(idx, 'lq')
        hq = self._load_image(idx, 'hq')
        lq_hsv = rgb2hsv(lq)
        lq_lab = rgb2lab(lq)
        sample = {'lq': lq, 'hq': hq, 'lq_hsv': lq_hsv, 'lq_lab': lq_lab}
        
        #resize
        if self.resize != 0:
            sample = self._resize(sample)

        #augmentation
        if self.if_aug is True:
            sample = self._augmentation(sample)

        #normalisation
        if self.if_norm is True:
            sample = self._norm(sample)
        
        #conver to pytorch tensor
        for key, value in sample.items():
            sample[key] = np2tensor(value)
        
        return sample
    
class EyeQSyntheticTestDataset(BaseDataset):
    #every 7 Synthetic low-quality image, has 1 high-quality image
    
    def __init__(self, cfgs):
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
        
        #image index list
        
        temp_namelist = os.listdir(self.hq_path)
        temp_namelist = [e.split('.')[0] for e in temp_namelist if '.png' in e]
        
        return temp_namelist
    

    def _load_image(self, idx, v):
        sampled_img = None
        
        if v == 'lq':
        #randomly choose degraded type
            r = random.randint(0, 15)
            self.this_r = r
            #print(f'type {self.de_type[r]} is selected')
            sampled_img = cv.imread(os.path.join(self.lq_path, \
                                                 self.namelist[idx] + '_'
                                                                    + str(r)
                                                                    + '.png'
                                                ))

            sampled_img = bgr2rgb(sampled_img)

            
        if v == 'hq':
        #directly return the high-quality image 
            sampled_img = cv.imread(os.path.join(self.hq_path, \
                                                 self.namelist[idx] + '.png'            
                                                ))
            
            sampled_img = bgr2rgb(sampled_img)
        
        if v == 'mask':
            sampled_img = cv.imread(os.path.join(self.mask_path, \
                                                 self.namelist[idx] + '.png'            
                                                ), 2)
        
        return sampled_img
        
    def __len__(self):
        return len(self.namelist)

    
    def __getitem__(self, idx):
        #print(f'{idx} is selected')
        lq = self._load_image(idx, 'lq')
        hq = self._load_image(idx, 'hq')
        mask = self._load_image(idx, 'mask')
        
        lq_hsv = rgb2hsv(lq)
        lq_lab = rgb2lab(lq)
        
        sample = {'lq': lq, 'hq': hq, 'mask': mask, 'lq_hsv': lq_hsv, 'lq_lab': lq_lab}
        
        #resize
        if self.resize != 0:
            sample = self._resize(sample)

        #normalisation
        if self.if_norm is True:
            sample = self._norm(sample)

        #conver to pytorch tensor
        for key, value in sample.items():
            sample[key] = np2tensor(value)
               
        return sample
    
    
class EyeQTestDataset(BaseDataset):
    #every 7 Synthetic low-quality image, has 1 high-quality image
    
    def __init__(self, cfgs):
        self.lq_path = os.path.join(cfgs.dataset_path, 'test', 'degraded_good', 'de_image')
        self.hq_path = os.path.join(cfgs.dataset_path, 'test', 'resized_good', 'image')
        self.mask_path = os.path.join(cfgs.dataset_path, 'test', 'resized_good', 'mask')
        self.json_path = os.path.join(cfgs.dataset_path, 'test', 'degraded_good', 'de_js_file')
 
        self.resize = 0
        self.if_norm = cfgs.if_norm

        if cfgs.resize == 0:
            self.resize = 0
        elif isinstance(cfgs.resize, list):
            self.resize = cfgs.resize
        else:
            self.resize = [cfgs.resize, cfgs.resize]
            
        self.df = pd.read_csv('./dataset/MIL_lq_test.csv')
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        #print(f'{idx} is selected')
        lq_name = self.df.loc[idx, 'image']
        hq_name = lq_name.split('_')
        hq_name = hq_name[0] + '_' + hq_name[1] + '.png'
        mask_name = hq_name
        
        lq = cv.imread(os.path.join(self.lq_path,  lq_name))
        lq = bgr2rgb(lq)    
        mask = cv.imread(os.path.join(self.mask_path,  mask_name), 2)
        hq = cv.imread(os.path.join(self.hq_path, hq_name))
        
        lq_hsv = rgb2hsv(lq)
        lq_lab = rgb2lab(lq)
        
        sample = {'lq': lq, 'hq': hq, 'mask': mask, 'lq_hsv': lq_hsv, 'lq_lab': lq_lab}
        
        #resize
        if self.resize != 0:
            sample = self._resize(sample)

        #normalisation
        if self.if_norm is True:
            sample = self._norm(sample)

        #conver to pytorch tensor
        for key, value in sample.items():
            sample[key] = np2tensor(value)
        
        
        sample['name'] = lq_name
        
        return sample
    
    
class EyeQUsableDataset(BaseDataset):
    def __init__(self, cfgs):
        self.lq_path = os.path.join(cfgs.dataset_path, 'test', 'resized_usable', 'image')
        self.mask_path = os.path.join(cfgs.dataset_path, 'test', 'resized_usable', 'mask')
        
        self.namelist = self._init_filename()
        #self.de_type = ['001', '010', '011', '100', '101', '110', '111']
        
        self.resize = 0
        self.if_norm = cfgs.if_norm

        if cfgs.resize == 0:
            self.resize = 0
        elif isinstance(cfgs.resize, list):
            self.resize = cfgs.resize
        else:
            self.resize = [cfgs.resize, cfgs.resize]
        
        self.vislist = []
    
    def _empty_vislist(self):
        self.vislist = []
        
    def _init_filename(self):
        '''
        image index list
        '''
        temp_namelist = os.listdir(self.lq_path)
        temp_namelist = [e.split('.')[0] for e in temp_namelist if '.png' in e]
        
        return temp_namelist
    

    def _load_image(self, idx, v):
        sampled_img = None
        
        if v == 'lq':
        #randomly choose degraded type
            sampled_img = cv.imread(os.path.join(self.lq_path, \
                                                 self.namelist[idx] + '.png'
                                                ))
            #self.vislist.append([self.namelist[idx], self.de_type[r]])
            
            sampled_img = (bgr2rgb(sampled_img), self.namelist[idx] + '.png')
        
        if v == 'mask':
            sampled_img = cv.imread(os.path.join(self.mask_path, \
                                                 self.namelist[idx] + '.png'            
                                                ), 2)
        
        return sampled_img
        
    def __len__(self):
        return len(self.namelist)
    
    def __getitem__(self, idx):
        #print(f'{idx} is selected')
        lq, file_name = self._load_image(idx, 'lq')
        mask = self._load_image(idx, 'mask')
        lq_hsv = rgb2hsv(lq)
        lq_lab = rgb2lab(lq)
        sample = {'lq': lq, 'lq_hsv': lq_hsv, 'lq_lab': lq_lab, 'mask': mask}
        
        #resize
        if self.resize != 0:
            sample = self._resize(sample)

        #normalisation
        if self.if_norm is True:
            sample = self._norm(sample)

        #conver to pytorch tensor
        for key, value in sample.items():
            sample[key] = np2tensor(value)
        
        sample['name'] = file_name
        
        return sample
    
    
class EyeQRejectDataset(BaseDataset):
    '''
    every 7 Synthetic low-quality image, has 1 high-quality image
    '''
    
    def __init__(self, cfgs):
        self.lq_path = os.path.join(cfgs.dataset_path, 'test', 'resized_reject', 'image')
        self.mask_path = os.path.join(cfgs.dataset_path, 'test', 'resized_reject', 'mask')
        
        self.namelist = self._init_filename()
        
        self.resize = 0
        self.if_norm = cfgs.if_norm

        if cfgs.resize == 0:
            self.resize = 0
        elif isinstance(cfgs.resize, list):
            self.resize = cfgs.resize
        else:
            self.resize = [cfgs.resize, cfgs.resize]
        
        self.vislist = []
    
    def _empty_vislist(self):
        self.vislist = []
        
    def _init_filename(self):
        '''
        image index list
        '''
        temp_namelist = os.listdir(self.lq_path)
        temp_namelist = [e.split('.')[0] for e in temp_namelist if '.png' in e]
        
        return temp_namelist
    

    def _load_image(self, idx, v):
        sampled_img = None
        
        if v == 'lq':
        #randomly choose degraded type
            sampled_img = cv.imread(os.path.join(self.lq_path, \
                                                 self.namelist[idx] + '.png'
                                                ))
            #self.vislist.append([self.namelist[idx], self.de_type[r]])
            
            sampled_img = (bgr2rgb(sampled_img), self.namelist[idx] + '.png')
        
        if v == 'mask':
            sampled_img = cv.imread(os.path.join(self.mask_path, \
                                                 self.namelist[idx] + '.png'            
                                                ), 2)
        
        return sampled_img
        
    def __len__(self):
        return len(self.namelist)
    
    def __getitem__(self, idx):
        #print(f'{idx} is selected')
        lq, file_name = self._load_image(idx, 'lq')
        mask = self._load_image(idx, 'mask')
        lq_hsv = rgb2hsv(lq)
        lq_lab = rgb2lab(lq)
        sample = {'lq': lq, 'lq_hsv': lq_hsv, 'lq_lab': lq_lab, 'mask': mask}
        
        #resize
        if self.resize != 0:
            sample = self._resize(sample)

        #normalisation
        if self.if_norm is True:
            sample = self._norm(sample)

        #conver to pytorch tensor
        for key, value in sample.items():
            sample[key] = np2tensor(value)
        
        sample['name'] = file_name
        
        return sample
    

class EyeQUnpairTrainDataset(BaseDataset):
    
    def __init__(self, cfgs):
        self.hq_path = os.path.join(cfgs.dataset_path, 'train', 'resized_good', 'image')
        self.lq_path = os.path.join(cfgs.dataset_path, 'train', 'resized_usable', 'image')
        self.hq_namelist = self._init_filename(self.hq_path)
        self.lq_namelist = self._init_filename(self.lq_path)
        
        self.if_aug = cfgs.if_aug
        self.resize = 0
        self.if_norm = cfgs.if_norm

        if cfgs.resize == 0:
            self.resize = 0
        elif isinstance(cfgs.resize, list):
            self.resize = cfgs.resize
        else:
            self.resize = [cfgs.resize, cfgs.resize]
    

    def _init_filename(self, path):
        '''
        image index list
        '''
        temp_namelist = os.listdir(path)
        temp_namelist = [e.split('.')[0] for e in temp_namelist if '.png' in e]
        
        return temp_namelist
    

    def _load_image(self, idx, v):
        sampled_img = None
        if v == 'lq':
        #randomly choose degraded type
            idx_lq = random.randint(0, len(self.lq_namelist) - 1)
            sampled_img = cv.imread(os.path.join(self.lq_path, \
                                                 self.lq_namelist[idx_lq]  + '.png'))
            
            sampled_img = bgr2rgb(sampled_img)
            
        if v == 'hq':
        #directly return the high-quality image 
            sampled_img = cv.imread(os.path.join(self.hq_path, \
                                                 self.hq_namelist[idx] + '.png'))
            
            sampled_img = bgr2rgb(sampled_img)
            
        return sampled_img
    
    def __len__(self):
        return len(self.hq_namelist)
    
    def __getitem__(self, idx):
        #print(f'{idx} is selected')
        hq = self._load_image(idx, 'hq')
        lq = self._load_image(idx, 'lq')
        lq_hsv = rgb2hsv(lq)
        lq_lab = rgb2lab(lq)
        sample = {'lq': lq, 'hq': hq, 'lq_hsv': lq_hsv, 'lq_lab': lq_lab}
        
        #resize
        if self.resize != 0:
            sample = self._resize(sample)

        #augmentation
        if self.if_aug is True:
            sample = self._augmentation(sample)

        #normalisation
        if self.if_norm is True:
            sample = self._norm(sample)
        
        #conver to pytorch tensor
        for key, value in sample.items():
            sample[key] = np2tensor(value)
        
        return sample
    
    
class EyeQTrainWUDataset(BaseDataset):
    '''
    every 7 low-quality image, has 1 high-quality image
    '''
    
    def __init__(self, cfgs):
        self.lq_path = os.path.join(cfgs.dataset_path, 'train', 'degraded_good', 'de_image')
        self.hq_path = os.path.join(cfgs.dataset_path, 'train', 'resized_good', 'image')
        self.usable_path = os.path.join(cfgs.dataset_path, 'train', 'resized_usable', 'image')
        self.usable_namelist = self._init_filename(self.usable_path)
        self.namelist = self._init_filename(self.hq_path)
        
        self.if_aug = cfgs.if_aug
        self.resize = 0
        self.if_norm = cfgs.if_norm

        if cfgs.resize == 0:
            self.resize = 0
        elif isinstance(cfgs.resize, list):
            self.resize = cfgs.resize
        else:
            self.resize = [cfgs.resize, cfgs.resize]
    

    def _init_filename(self, path):
        '''
        image index list
        '''
        temp_namelist = os.listdir(path)
        temp_namelist = [e.split('.')[0] for e in temp_namelist if '.png' in e]
        
        return temp_namelist
    

    def _load_image(self, idx, v):
        sampled_img = None
        if v == 'lq':
        #randomly choose degraded type
            r = random.randint(0, 15)
                
            #print(f'type {self.de_type[r]} is selected')
            sampled_img = cv.imread(os.path.join(self.lq_path, \
                                                 self.namelist[idx] + '_'
                                                                    + str(r)
                                                                    + '.png'
                                                ))
            
            sampled_img = bgr2rgb(sampled_img)
            
        if v == 'hq':
        #directly return the high-quality image 
            sampled_img = cv.imread(os.path.join(self.hq_path, \
                                                 self.namelist[idx] + '.png'            
                                                ))
            
            sampled_img = bgr2rgb(sampled_img)
        
        if v == 'usable':
        #directly return the high-quality image 
            idx_u = random.randint(0, len(self.usable_namelist) - 1)
            sampled_img = cv.imread(os.path.join(self.usable_path, \
                                                 self.usable_namelist[idx_u] + '.png'            
                                                ))
            
            sampled_img = bgr2rgb(sampled_img)
        
        return sampled_img
    
        
    
    def __len__(self):
        return len(self.namelist)
    
    def __getitem__(self, idx):
        #print(f'{idx} is selected')
        lq = self._load_image(idx, 'lq')
        hq = self._load_image(idx, 'hq')
        usable = self._load_image(idx, 'usable')
        lq_hsv = rgb2hsv(lq)
        lq_lab = rgb2lab(lq)
        usable_hsv = rgb2hsv(usable)
        usable_lab = rgb2lab(usable)
        sample = {'lq': lq, 'hq': hq, 'lq_hsv': lq_hsv, 'lq_lab': lq_lab, 
                 'usable': usable, 'usable_hsv': usable_hsv, 'usable_lab': usable_lab}
        
        #resize
        if self.resize != 0:
            sample = self._resize(sample)

        #augmentation
        if self.if_aug is True:
            sample = self._augmentation(sample)

        #normalisation
        if self.if_norm is True:
            sample = self._norm(sample)
        
        #conver to pytorch tensor
        for key, value in sample.items():
            sample[key] = np2tensor(value)
        
        return sample