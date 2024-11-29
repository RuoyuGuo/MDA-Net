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

class EyeQTrainDataset(Dataset):
    '''
    every 7 low-quality image, has 1 high-quality image
    '''
    
    def __init__(self, cfgs):
        self.lq_path = os.path.join(cfgs.dataset_path, 'train', 'degraded_good', 'de_image')
        self.hq_path = os.path.join(cfgs.dataset_path, 'train', 'resized_good', 'image')
        self.json_path = os.path.join(cfgs.dataset_path, 'train', 'degraded_good', 'de_js_file')
        self.namelist = self._init_filename()
      
        self.de2num = {'001': 0,
                       '010': 1,
                       '011': 2, 
                       '100': 3,
                       '101': 4,
                       '110': 5,
                       '111': 6,}
        
        #self.transform = RandomResizedCropCoord(cfgs.aug_resize)
        
        if cfgs.seed is not None:
            self.only_ill_rng = np.random.default_rng(cfgs.seed)
        else:
            self.only_ill_rng = None
        
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
            jsonfile = json.load(open(os.path.join(self.json_path, \
                                            self.namelist[idx] + '_'
                                                               + str(r)
                                                               + '.json'), 'r'))   
            detype = self.de2num[jsonfile['type']]
        
            # sampled_img = [bgr2rgb(sampled_img), detype]
            
            #print(f'sample {self.namelist[idx]}_{self.de_type[r]}.png')
            #print(f'Also sample {self.namelist[idx]}_{self.de_type_ir[r][r_ir]}.png')
            #print()
                        
            r_v1, r_v2 =  random.sample([i for i in range(16) if i != r], k=2)
            
            s_img_view1 = cv.imread(os.path.join(self.lq_path, \
                                                 self.namelist[idx] + '_'
                                                                    + str(r_v1)
                                                                    + '.png'))
                                                 
            s_img_view2 = cv.imread(os.path.join(self.lq_path, \
                                         self.namelist[idx] + '_'
                                                            + str(r_v2)
                                                            + '.png')) 
            
            sampled_img = (bgr2rgb(sampled_img), detype, bgr2rgb(s_img_view1), bgr2rgb(s_img_view2))
            
        if v == 'hq':
        #directly return the high-quality image 
            sampled_img = cv.imread(os.path.join(self.hq_path, \
                                                 self.namelist[idx] + '.png'            
                                                ))
            
            sampled_img = bgr2rgb(sampled_img)
                
        if v == 'r_hq':
        #return a random hq image for training
            r = random.randint(0, len(self.namelist)-1)
            sampled_img = cv.imread(os.path.join(self.hq_path, \
                                                 self.namelist[r] + '.png'            
                                                ))
            
            sampled_img = bgr2rgb(sampled_img)
            
        return sampled_img


    def _flip(self, sample):
        '''
        flip function
        '''
        for key in sample:
            sample[key] = np.fliplr(sample[key])
        
        return sample


    def _rot(self, sample, r_rot):
        '''
        rotate image
        '''

        #90 degree
        if r_rot >= 1:
             for key in sample:
                sample[key] = np.rot90(sample[key])           

        #180 degree
        if r_rot >= 2:
             for key in sample:
                sample[key] = np.rot90(sample[key])  
        
        #270 degree
        if r_rot >= 3:
             for key in sample:
                sample[key] = np.rot90(sample[key])  

        return sample
        

    def _augmentation(self, sample):
        '''
        apply a series of augmentation 
        currently only horizontal flip
        '''

        r_flip = random.randint(0,1)
        r_rot = random.randint(0,3)
        #print(f'aug method: {r_flip} and {r_rot}')

        if r_flip == 1:
            sample = self._flip(sample)

        if r_rot != 0:
            sample = self._rot(sample, r_rot)

        return sample

    
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
        lq, de_r, lq_v1, lq_v2 = self._load_image(idx, 'lq')
        hq = self._load_image(idx, 'hq')
        r_hq = self._load_image(idx, 'r_hq')
        #view1, view2 = self._load_image(idx, 'augv')
        sample = {'lq': lq, 'hq': hq, 'r_hq': r_hq, 'lq_v1': lq_v1, 'lq_v2': lq_v2}
        #sample = {'lq': lq, 'hq': hq, 'view1': view1, 'view2': view2}
        
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
        
        return sample['lq'], sample['hq'], de_r, sample['r_hq'], sample['lq_v1'], sample['lq_v2']
    
class EyeQTrainWUDataset(Dataset):
    '''
    every 7 low-quality image, has 1 high-quality image
    '''
    
    def __init__(self, cfgs):
        self.lq_path = os.path.join(cfgs.dataset_path, 'train', 'degraded_good', 'de_image')
        self.hq_path = os.path.join(cfgs.dataset_path, 'train', 'resized_good', 'image')
        self.usable_path = os.path.join(cfgs.dataset_path, 'train', 'resized_usable', 'image')
        self.usable_list = os.listdir(self.usable_path)
        self.usable_list = [e.split('.')[0] for e in self.usable_list if '.png' in e]
        self.json_path = os.path.join(cfgs.dataset_path, 'train', 'degraded_good', 'de_js_file')
        self.namelist = self._init_filename()
      
        self.de2num = {'001': 0,
                       '010': 1,
                       '011': 2, 
                       '100': 3,
                       '101': 4,
                       '110': 5,
                       '111': 6,}
        
        #self.transform = RandomResizedCropCoord(cfgs.aug_resize)
        
        if cfgs.seed is not None:
            self.only_ill_rng = np.random.default_rng(cfgs.seed)
        else:
            self.only_ill_rng = None
        
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
            jsonfile = json.load(open(os.path.join(self.json_path, \
                                            self.namelist[idx] + '_'
                                                               + str(r)
                                                               + '.json'), 'r'))   
            detype = self.de2num[jsonfile['type']]
        
            # sampled_img = [bgr2rgb(sampled_img), detype]
            
            #print(f'sample {self.namelist[idx]}_{self.de_type[r]}.png')
            #print(f'Also sample {self.namelist[idx]}_{self.de_type_ir[r][r_ir]}.png')
            #print()
                        
            r_v1, r_v2 =  random.sample([i for i in range(16) if i != r], k=2)
            
            s_img_view1 = cv.imread(os.path.join(self.lq_path, \
                                                 self.namelist[idx] + '_'
                                                                    + str(r_v1)
                                                                    + '.png'))
                                                 
            s_img_view2 = cv.imread(os.path.join(self.lq_path, \
                                         self.namelist[idx] + '_'
                                                            + str(r_v2)
                                                            + '.png')) 
            
            sampled_img = (bgr2rgb(sampled_img), detype, bgr2rgb(s_img_view1), bgr2rgb(s_img_view2))
            
        if v == 'hq':
        #directly return the high-quality image 
            sampled_img = cv.imread(os.path.join(self.hq_path, \
                                                 self.namelist[idx] + '.png'            
                                                ))
            
            sampled_img = bgr2rgb(sampled_img)
                
        if v == 'r_hq':
        #return a random hq image for training
            r = random.randint(0, len(self.namelist)-1)
            sampled_img = cv.imread(os.path.join(self.hq_path, \
                                                 self.namelist[r] + '.png'            
                                                ))
            
            sampled_img = bgr2rgb(sampled_img)
        
        if v == 'usable':
            sampled_img = cv.imread(os.path.join(self.usable_path, \
                                                 self.usable_list[idx%len(self.usable_list)] + '.png'   
                                                ))
            
            sampled_img = bgr2rgb(sampled_img)
        
        return sampled_img


    def _flip(self, sample):
        '''
        flip function
        '''
        for key in sample:
            sample[key] = np.fliplr(sample[key])
        
        return sample


    def _rot(self, sample, r_rot):
        '''
        rotate image
        '''

        #90 degree
        if r_rot >= 1:
             for key in sample:
                sample[key] = np.rot90(sample[key])           

        #180 degree
        if r_rot >= 2:
             for key in sample:
                sample[key] = np.rot90(sample[key])  
        
        #270 degree
        if r_rot >= 3:
             for key in sample:
                sample[key] = np.rot90(sample[key])  

        return sample
        

    def _augmentation(self, sample):
        '''
        apply a series of augmentation 
        currently only horizontal flip
        '''

        r_flip = random.randint(0,1)
        r_rot = random.randint(0,3)
        #print(f'aug method: {r_flip} and {r_rot}')

        if r_flip == 1:
            sample = self._flip(sample)

        if r_rot != 0:
            sample = self._rot(sample, r_rot)

        return sample

    
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
        lq, de_r, lq_v1, lq_v2 = self._load_image(idx, 'lq')
        hq = self._load_image(idx, 'hq')
        r_hq = self._load_image(idx, 'r_hq')
        usable = self._load_image(idx, 'usable')
        #view1, view2 = self._load_image(idx, 'augv')
        sample = {'lq': lq, 'hq': hq, 'r_hq': r_hq, 'lq_v1': lq_v1, 'lq_v2': lq_v2, 'usable': usable}
        #sample = {'lq': lq, 'hq': hq, 'view1': view1, 'view2': view2}
        
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
        
        return sample['lq'], sample['hq'], de_r, sample['r_hq'], sample['lq_v1'], sample['lq_v2'], sample['usable']
    
    
class EyeQSyntheticTestDataset(Dataset):
    #every 7 Synthetic low-quality image, has 1 high-quality image
    
    def __init__(self, cfgs):
        self.lq_path = os.path.join(cfgs.dataset_path, 'test', 'degraded_good', 'de_image')
        self.hq_path = os.path.join(cfgs.dataset_path, 'test', 'resized_good', 'image')
        self.mask_path = os.path.join(cfgs.dataset_path, 'test', 'resized_good', 'mask')
        
        self.json_path = os.path.join(cfgs.dataset_path, 'test', 'degraded_good', 'de_js_file')
        self.de2num = {'001': 0,
                       '010': 1,
                       '011': 2, 
                       '100': 3,
                       '101': 4,
                       '110': 5,
                       '111': 6,}
        
        
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
            jsonfile = json.load(open(os.path.join(self.json_path, \
                                            self.namelist[idx] + '_'
                                                               + str(r)
                                                               + '.json'), 'r'))   
            detype = self.de2num[jsonfile['type']]
        

            sampled_img = (bgr2rgb(sampled_img), detype)
            
            
            
            
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
        lq, detype = self._load_image(idx, 'lq')
        hq = self._load_image(idx, 'hq')
        mask = self._load_image(idx, 'mask')
        
       # name = int(self.namelist[idx].split('_')[0])
        #de_r = self.this_r
        sample = {'lq': lq, 'hq': hq, 'mask': mask}
        
        #resize
        if self.resize != 0:
            sample = self._resize(sample)

        #normalisation
        if self.if_norm is True:
            sample = self._norm(sample)

        #conver to pytorch tensor
        for key, value in sample.items():
            sample[key] = np2tensor(value)
        
        
        return sample['lq'], sample['hq'], sample['mask'], detype#, name#, de_r#,  

    
class EyeQTestDataset(Dataset):
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
        
    def _norm(self, sample):
        for key in sample:
            sample[key] = sample[key]/255.0

        return sample
    
    def _resize(self, sample):
        for key in sample:
            sample[key] = cv.resize(sample[key], self.resize)  
        
        return sample    
        
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
        
        return sample['lq'], sample['mask'], lq_name#, de_r#,  
    
    
class EyeQUsableDataset(Dataset):
    '''
    every 7 Synthetic low-quality image, has 1 high-quality image
    '''
    
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
        lq, file_name = self._load_image(idx, 'lq')
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
        
        return sample['lq'], sample['mask'], file_name
    

class EyeQRejectDataset(Dataset):
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
        lq, file_name = self._load_image(idx, 'lq')
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
        
        return sample['lq'], sample['mask'], file_name