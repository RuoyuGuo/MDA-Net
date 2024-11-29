import os
import torch
import cv2 as cv
from copy import deepcopy

from utils.utils import *


class MYLog():
    def __init__(self, dataset, cfgs=None, comments=None, val=None, test=None, mode='train'):
        os.makedirs('./myoutput', exist_ok=True)

        #init path name
        self.path               = self._init_path()
        self.dataset            = dataset

        self.training_cfgs      = os.path.join(self.path, self.dataset, 'training_config.txt')
        self.training_log       = os.path.join(self.path, self.dataset, 'training_log.txt')

        if val is not None:
            self.val_log            = os.path.join(self.path, self.dataset, 'val_log.txt')
            self.val_output_path    = os.path.join(self.path, self.dataset, 'output', 'val')
        if test is not None:
            self.test_log            = os.path.join(self.path, self.dataset, 'test_log.txt')
            self.test_output_path    = os.path.join(self.path, self.dataset, 'output', 'test')

        self.network_cp_path    = os.path.join(self.path, self.dataset, 'checkpoint')
        
        if mode == 'train':
            #create folder
            if val is not None:
                os.makedirs(self.val_output_path, exist_ok=True)
            if test is not None:
                os.makedirs(self.test_output_path, exist_ok=True)
            os.makedirs(self.network_cp_path)

            #create file
            f = open(self.training_cfgs, 'w')
            f.close()
            f = open(self.training_log, 'w')
            f.close()

            if val is not None:
                f = open(self.val_log, 'w')
                f.close()

            if test is not None:
                f = open(self.test_log, 'w')
                f.close()

            #initial log
            if cfgs is not None:
                self.log_cfgs(cfgs)

            if comments is not None:
                self.log_comments(comments)

    def _num2fold(self, x):
        return str(x).zfill(5)


    def _init_path(self, mode='train'):
        if mode == 'train':
            fold_list = os.listdir('./myoutput')
            fold_name = 0

            while self._num2fold(fold_name) in fold_list:            
                fold_name += 1

            path_name = os.path.join('./myoutput', self._num2fold(fold_name))
            os.makedirs(path_name, exist_ok=True)
        else:
            path_name = os.path.join('./myoutput', self._num2fold(mode))
            
        return path_name         


    def log_comments(self, s):
        with open(self.training_cfgs, 'a') as f:
            f.write(s+'\n')
            print(s)


    def log_cfgs(self, a):
        if not isinstance(a, dict):
            raise RuntimeError('Only dict type cfgs could be logged.')

        max_len_key = 0
        for key in a.keys():
            if len(key) > max_len_key:
                max_len_key = len(key)
        max_len_key += 4

        max_len_value = 0
        for value in a.values():
            if len(str(value)) > max_len_value:
                max_len_value = len(key)

        with open(self.training_cfgs, 'a') as f:
            print('Training configurations:')
            print()
            for key, value in a.items():
                s = f'{key:<{max_len_key}}: {str(value):<{max_len_value}}'
                f.write(s+'\n')
                print(s)


    def log(self, s, stage):
        if not isinstance(s, str):
            raise RuntimeError('only string can be logged.')

        if stage == 'train':
            with open(self.training_log, 'a') as f:
                f.write(s+'\n')
                print(s)
        
        if stage == 'val':
            with open(self.val_log, 'a') as f:
                f.write(s+'\n')
                print(s)


    def save_net(self, netdict, netname):
        torch.save(deepcopy(netdict), os.path.join(self.network_cp_path, netname))
        print('Save net')


    def log_output(self, img, name, stage):
        if len(img.shape) == 4:
            if img.shape[0] != 1:
                raise RuntimeError('only one image can be saved')
            else:
                img = img[0]

        if isinstance(img, torch.Tensor):
            saved = f2u8(rgb2bgr(tensor2np(img)))
        else:
            saved = f2u8(rgb2bgr(img))


        if stage == 'val':
            cv.imwrite(os.path.join(self.val_output_path, name), saved)
        
        if stage == 'test':
            cv.imwrite(os.path.join(self.test_output_path, name), saved)