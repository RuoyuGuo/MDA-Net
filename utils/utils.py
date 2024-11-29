import torch
import numpy as np
import cv2 as cv
import torch.nn.functional as F
import torch.nn as nn
import random
import math

from torchvision.transforms import functional as transF
from torch.utils.data import Dataset

class BaseDataset(Dataset):
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

def format_time(s):
    s = round(s)

    #s
    if s < 60:
        return f'{s}s'
    #m s
    elif s < 60 * 60:
        return f'{s // 60:02}m {s % 60:02}s'
    #h m s
    elif s < 60 * 60 * 24:
        return f'{s // (60*60):02}h {(s // 60) % 60 :02}m {s % 60:02}s'
    #d h m
    else:
        return f'{s // (24 * 60 * 60)}d {(s // (60 * 60)) % 24:02}h {(s // 60) % 60}m'
    

def np2tensor(data):
    '''
    H * W : numpy array -> 1 * H * W : torch tensor
    H * W * C : numpy array -> C * H * W : torch tensor
    B * H * W * C : numpy array -> B * C * H * W : torch tensor
    '''

    #gray scale
    if len(data.shape) == 2:
        data = data[:,:, np.newaxis]
        return torch.from_numpy(np.ascontiguousarray(np.transpose(data, (2,0,1)))).float()
    #RGB
    elif len(data.shape) == 3:
        return torch.from_numpy(np.ascontiguousarray(np.transpose(data, (2,0,1)))).float()
    #Batch Image
    elif len(data.shape) == 4:
        return torch.from_numpy(np.ascontiguousarray(np.transpose(data, (0,3,1,2)))).float()
    else:
        raise RuntimeError(f'Wrong Input dimension, {data.shape}')


def tensor2np(data):
    '''
    C * H * W : torch tensor -> H * W * C : numpy array 

    B * C * H * W : torch tensor -> B * H * W * C : numpy array 
    '''

    data = data.cpu().detach()

    #CHW
    if len(data.shape) == 2 or len(data.shape) == 3:
        return data.permute(1,2,0).numpy()
    #BCHW
    elif len(data.shape) == 4:
        return data.permute(0,2,3,1).numpy()
    else:
        raise RuntimeError(f'Wrong Input dimension, {data.shape}')


def bgr2rgb(data):
    '''
    BGR -> RGB channel
    '''
    return cv.cvtColor(data, cv.COLOR_BGR2RGB)


def rgb2bgr(data):
    '''
    BGR -> RGB channel
    '''
    return cv.cvtColor(data, cv.COLOR_RGB2BGR)


def f2u8(img):
    img = img * 255
    img = np.clip(img, 0, 255)

    return img.astype(np.uint8)


class RandomResizedCropCoord(object):
    """Crop the given PIL Image to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.7, 1.0), ratio=(3. / 4., 4. / 3.)):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        #self.interpolation = A.
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = img.shape[1], img.shape[2]
        area = height * width

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w, height, width

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w, height, width

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        
        
        i, j, h, w, height, width = self.get_params(img, self.scale, self.ratio)
        coord = torch.Tensor([float(j) / (width - 1), float(i) / (height - 1),
                              float(j + w - 1) / (width - 1), float(i + h - 1) / (height - 1)])
        
        #print(i, j, h, w, height, width)
        
        #print(coord.shape)
        return transF.resized_crop(img, i, j, h, w, self.size), coord
    

def pixel_shuffle_down_sampling(x:torch.Tensor, f:int, pad:int=0, pad_value:float=0.):
    '''
    pixel-shuffle down-sampling (PD) from "When AWGN-denoiser meets real-world noise." (AAAI 2019)
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad between each down-sampled images
        pad_value (float) : padding value
    Return:
        pd_x (Tensor) : down-shuffled image tensor with pad or not
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,3,2,4).reshape(c, w+2*f*pad, h+2*f*pad)
    # batched image tensor
    else:
        b,c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(b,c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,2,4,3,5).reshape(b,c,w+2*f*pad, h+2*f*pad)

def pixel_shuffle_up_sampling(x:torch.Tensor, f:int, pad:int=0):
    '''
    inverse of pixel-shuffle down-sampling (PD)
    see more details about PD in pixel_shuffle_down_sampling()
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad will be removed
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        before_shuffle = x.view(c,f,w//f,f,h//f).permute(0,1,3,2,4).reshape(c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)   
    # batched image tensor
    else:
        b,c,w,h = x.shape
        before_shuffle = x.view(b,c,f,w//f,f,h//f).permute(0,1,2,4,3,5).reshape(b,c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)
    

def pixel_down_shuffle(x, ps):
    '''
    down sample image
    doesn't require computation
    
    [B], C, W, H --> [B|1] * ps * ps, C, W//ps, H//ps

    Parameters:
    ---------------------------
    @x   : Input image (Tensor)
    @ps  : downsample factor

    '''

    #c, w, h
    if len(x.shape) == 3:
        x = x.unsqeueeze(0)
    #b, c, w, h
    if len(x.shape) != 4:
        raise RuntimeError(f'Input tensor should be b,c,w,h, or c,w,h but have {x.shape}')

    b, c, w, h = x.shape

    shuffled_0 = F.pixel_unshuffle(x, ps)
    shuffled_1 = shuffled_0.view(b, c, ps, ps, w//ps, h//ps)\
                            .permute(0, 2, 3, 1, 4, 5)\
                            .reshape(b*ps*ps, c, w//ps, h//ps)

    return shuffled_1


def pixel_up_shuffle(x, ps):
    '''
    up sample image
    doesn't require computation
    
    [B|1] * ps * ps, C, W//ps, H//ps --> [B], C, W, H

    Parameters:
    ---------------------------
    @x   : Input image (Tensor)
    @ps  : downsample factor

    '''

    #b, c, w, h
    if len(x.shape) != 4:
        raise RuntimeError(f'Input tensor should be b,c,w,h but have {x.shape}')

    b, c, w, h = x.shape

    inv_shuffled_1 = x.view(b//ps//ps, ps, ps, c, w, h)\
                            .permute(0, 3, 1, 2, 4, 5)\
                            .reshape(b//ps//ps, c*ps*ps, w, h)
    #print(inv_shuffled_1.shape)
    inv_shuffled_0 = F.pixel_shuffle(inv_shuffled_1, ps)

    return inv_shuffled_0

def rgb2hsv(img):
    return cv.cvtColor(img, cv.COLOR_RGB2HSV)

def rgb2lab(img):
    return cv.cvtColor(img, cv.COLOR_RGB2Lab)