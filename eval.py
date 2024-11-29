import os
import random
import time

import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
from torch.utils.data import DataLoader
from dotmap import DotMap
from PIL import Image
from tqdm import tqdm

from network import DANet
from utils.logger import MYLog
from utils.utils import format_time, tensor2np
from utils.metrics import eval_metrics as my_metrics
from utils.myloss import GANLoss


import argparse

class EnhancementFW():
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.generator = self.net_init(framework=cfgs.framework)
    
    def net_init(self, framework):
        MODEL_SELECTION = {   'baseline' : DANet.pix2pix,
                                'dfm'      : DANet.pix2pixWdfm,   
                                'clf'      : DANet.pix2pixWdfmW2clf,
                    }
    
        network = None
        network = MODEL_SELECTION[framework]()
        print()
        print('Training backbone: pix2pix...')
        print('Framework: ', end='')
        print(network)
        print()
            
        time.sleep(1)
        return network.to('cuda')

    def load(self,path):
        network_checkpoint = torch.load(path)
        self.generator.load_state_dict(network_checkpoint['gen'])

    def train(self):
        self.generator.train()
        
    def eval(self):
        self.generator.eval()

    def forward(self, inputs, flag_clf_clean):
        outputs = self.generator(inputs, flag_clf_clean)
        return outputs
        
class myModel():
    def __init__(self, cfgs):
        self.iteration = 0
        self.total_time = 0
        self.total_iteration = 0
        self.cfgs = cfgs
        
        self.model = EnhancementFW(cfgs)      
        self.SINCE_TIME = 0 
        
    
    def load(self, net_index, net_epoch):
        cp_path = os.path.join('./myoutput', \
                               str(net_index).zfill(5), \
                               'EyeQ', \
                               'checkpoint', \
                               'danet_' + str(net_epoch)+'_net.pt')
        self.model.load(cp_path)
        
        print(f'Loading net on {net_index} and {net_epoch}.')
        print()
    
    def eval(self, dataloader):
        print('Evaluation...')
        print()
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(dataloader):
                lq_img, mask, filename = batch
                lq_img, mask = lq_img.to('cuda'), mask.to('cuda')
                
                fake_hq_imgs, _ = self.model.forward((lq_img, None), False)

                
                img = fake_hq_imgs.squeeze(0).permute(1,2,0).cpu().numpy() * 255
                img = np.clip(img, 0, 255).astype(np.uint8)
                #img.save(os.path.join(cfgs.output_path, filename[0]))
                cv.imwrite(os.path.join(cfgs.output_path, str(filename[0])), img[:,:,::-1])
                
if __name__ == '__main__':
    torch.manual_seed(2023)
    torch.cuda.manual_seed(2023)
    np.random.seed(2023)
    random.seed(2023)

    parser = argparse.ArgumentParser(description='Backbone and framework selection.')
    parser.add_argument('--framework', type=str, required=True,
                    help='Selection framework: [baseline, dfm, clf]]')
    

    parser.add_argument('--load_index', type=int, required=True,
                    help='load pretrain model, 0 for training from scratch')    
    parser.add_argument('--load_epoch', type=int, required=True,
                    help='load pretrain model, 0 for training from scratch')    
    
    #path
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True,
                    help='dataset path')    
    parser.add_argument('--output_path', type=str, required=True,
                    help='save path')    
      
    args = parser.parse_args()

    cfgs = {'resize'    : 256, 
            'if_aug'    : False, 
            'if_norm'   : True,
            'init_lr'   : 0.0001,                   # learning rate
            'dataset_path'      : args.dataset_path, 
            'dataset'           : args.dataset, 
            'output_path'       : args.output_path,
            
            #training detial
            'beta1'             : 0.9,                # 0 or 0.9
            'beta2'             : 0.9,
            'cp_interval'       : 5, 
            'eval_interval'     : 10,
            'seed'              : 2023,
            'eta_min'           : 1e-7,
            'gan_loss'          : 'nsgan',
            
            #loss weight
            're_loss_w'         : 1.5,
            'clf_loss_w'        : 1.5,
            'gan_loss_w'        : 0.1,
            'd2g_lr'            : 0.1,
            'sim_loss_w'        : 1,
            
            #framework
            'load_index'        : args.load_index,
            'load_epoch'        : args.load_epoch,
            'framework'         : args.framework,       #baseline < dfm < clf
            }

    comments = ''
    cfgs = DotMap(cfgs, _dynamic=False)
        
    if cfgs.dataset == 'eyeqtest':
        from utils.eyeq_dataset import EyeQTestDataset as TestDataset
    if cfgs.dataset == 'eyequsable':
        from utils.eyeq_dataset import EyeQUsableDataset as TestDataset
    if cfgs.dataset == 'eyeqreject':
        from utils.eyeq_dataset import EyeQRejectDataset as TestDataset
    if cfgs.dataset == 'drive':
        from utils.drive_dataset import DRIVETestDataset as TestDataset
    if cfgs.dataset == 'refuge':
        from utils.refuge_dataset import REFUGETestDataset as TestDataset
            
    testloader = DataLoader(TestDataset(cfgs), batch_size=1, shuffle=False)
    
    DANetformer = myModel(cfgs)
    DANetformer.load(cfgs.load_index, cfgs.load_epoch)
    DANetformer.eval(testloader)
