import torch
import numpy as np

from utils.utils import tensor2np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class eval_metrics():
    #metrics for 
    
    def __init__(self, disable_fun=False):

        self.psnr   = 0
        self.ssim   = 0
        self.count  = 0
        self.disable_fun = disable_fun
        
    def batchpsnr(self, bimg1, bimg2):
        '''
        a batch of image
        
        bimg1: B * C * H * W
        bimg2: B * C * H * W
        '''

        if bimg1.shape != bimg2.shape:
            raise RuntimeError(f'two image batch should have same shape. A: {bimg1.shape} B: {bimg2.shape}')
        if len(bimg1.shape) != 4:
            raise RuntimeError('should be a batch of image')

        sum_psnr    = 0
        num         = bimg1.shape[0]

        for i in range(num):
            sum_psnr += psnr(bimg1[i], bimg2[i])

        return sum_psnr/num

    def batchssim(self, bimg1, bimg2):
        '''
        a batch of image
        
        bimg1: B * C * H * W
        bimg2: B * C * H * W
        '''

        if bimg1.shape != bimg2.shape:
            raise RuntimeError(f'two image batch should have same shape. A: {bimg1.shape} B: {bimg2.shape}')
        if len(bimg1.shape) != 4:
            raise RuntimeError('should be a batch of image')

        sum_ssim    = 0
        num         = bimg1.shape[0]

        for i in range(num):
            sum_ssim += ssim(bimg1[i], bimg2[i])

        return sum_ssim/num

    def items(self):
        if self.disable_fun is not True:
            return [('PSNR', self.psnr), ('SSIM', self.ssim)]
    
    def op(self, x, y):
        if self.disable_fun is not True:
            assert len(x.shape) == 4
            assert len(y.shape) == 4

            result_batch_psnr = self.batchpsnr(x, y)
            result_batch_ssim = self.batchssim(x, y)
            num = x.shape[0]

            self.psnr += num * result_batch_psnr
            self.ssim += num * result_batch_ssim
            self.count += num
                
            return result_batch_psnr, result_batch_ssim
    
    def final(self):
        if self.disable_fun is not True:
            self.psnr /= self.count
            self.ssim /= self.count

    def empty(self):
        if self.disable_fun is not True:
            self.psnr = 0
            self.ssim = 0
            self.count = 0



def psnr(img1, img2):
    '''
    image value range : [0-1]
    clipping for model output
    '''

    # tensor to numpy
    if isinstance(img1, torch.Tensor):
        img1 = tensor2np(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor2np(img2)

    # numpy value cliping & chnage type to uint8
    img1 = np.clip(img1, 0, 1.0)
    img2 = np.clip(img2, 0, 1.0)

    return peak_signal_noise_ratio(img1, img2, data_range=1.0)

def ssim(img1, img2):
    '''
    image value range : [0 - 1]
    clipping for model output
    '''

    # tensor to numpy
    if isinstance(img1, torch.Tensor):
        img1 = tensor2np(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor2np(img2)

    # numpy value cliping
    img1 = np.clip(img1, 0, 1.0)
    img2 = np.clip(img2, 0, 1.0)

    return structural_similarity(img1, img2, channel_axis=2, data_range=1.0)