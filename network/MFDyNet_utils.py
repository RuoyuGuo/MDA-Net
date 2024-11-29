import torch
import torch.nn as nn
from network.kpn_network import KernelConv
import torch.nn.functional as F
from einops import rearrange

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
    
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
                
                #print('init layer', classname)
                
            elif classname.find('BatchNorm2d') != -1:
                #print('Init norm', classname)
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
    

class Downsample3(nn.Module):
    def __init__(self, c):
        super(Downsample3, self).__init__()
        self.net1 = Downsample(c)
        self.net2 = Downsample(c)
        self.net3 = Downsample(c)
    
    def forward(self, x_rgb, x_hsv, x_lab):
        x_rgb = self.net1(x_rgb)
        x_hsv = self.net2(x_hsv)
        x_lab = self.net3(x_lab)
        
        return x_rgb, x_hsv, x_lab

class down(nn.Module):
    def __init__(self, c):
        super(down, self).__init__()
        self.net1 = nn.Conv2d(c, c*2, 3, 1, 1, padding_mode='reflect')
        
        self.maxpool = nn.MaxPool2d(2)
    
    def forward(self, x_rgb):
        x_rgb = self.maxpool(self.net1(x_rgb))
        
        return x_rgb

class up(nn.Module):
    def __init__(self, c):
        super(up, self).__init__()
        self.net = nn.Conv2d(c, c//2, 3, 1, 1, padding_mode='reflect')
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
    
    def forward(self, x):
        return self.up(self.net(x))
    

class ConvReLUBlock(nn.Module):
    def __init__(self, inc, outc):
        super(ConvReLUBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(inc, outc, 1, 1, 0, padding_mode='reflect')
        self.conv3x3 = nn.Conv2d(outc, outc, 3, 1, 1, padding_mode='reflect')
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.conv3x3(self.conv1x1(x)))
    
    
class ConvReLU(nn.Module):
    def __init__(self, inc, outc, kernel, stride=1, padding=0):
        super(ConvReLU, self).__init__()
        
        self.conv = nn.Conv2d(inc, outc, kernel, stride, padding, padding_mode='reflect')
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.conv(x))
    
    
class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        return out       

    
#Blocks design
class MFBlock(nn.Module):
    def __init__(self, c, r=4):
        super(MFBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.global_s = ConvReLU(c, c//r, 1)
        self.rgb_e = nn.Conv2d(c//r, c, 1, 1)
        self.hsv_e = nn.Conv2d(c//r, c, 1, 1)
        self.lab_e = nn.Conv2d(c//r, c, 1, 1)
        self.project  = nn.Conv2d(c, c, 1)
        
    def forward(self, x_rgb, x_hsv, x_lab):
        global_select = self.gap(x_rgb + x_hsv + x_lab)
        global_select = self.global_s(global_select)
        rgb_select    = self.rgb_e(global_select).squeeze(-1).squeeze(-1).unsqueeze(1)
        hsv_select    = self.hsv_e(global_select).squeeze(-1).squeeze(-1).unsqueeze(1)
        lab_select    = self.lab_e(global_select).squeeze(-1).squeeze(-1).unsqueeze(1)

        total_select = torch.cat([rgb_select, hsv_select, lab_select], dim=1)
        total_select = torch.softmax(total_select, dim=1)
      
        select_rgb = x_rgb * (total_select[:, 0]).unsqueeze(-1).unsqueeze(-1)
        select_hsv = x_hsv * (total_select[:, 1]).unsqueeze(-1).unsqueeze(-1)
        select_lab = x_lab * (total_select[:, 2]).unsqueeze(-1).unsqueeze(-1)
        
        select_feature = self.project(select_rgb+select_hsv+select_lab)

        return select_feature  
    

class DyBlock(nn.Module):
    def __init__(self, c):
        super(DyBlock, self).__init__()
        self.proj1 = nn.Conv2d(c, c, 1)
        self.proj2 = nn.Conv2d(c, c, 1)
        
        self.kernel_conv = nn.Sequential(nn.Conv2d(c, c*3, 1),
                                         nn.Conv2d(c*3, c*3, 3, 1, 1, groups=c),)
    
    def forward(self, x, x_mf):
        kernel = self.proj1(x) * self.proj2(x_mf)
        kernel = self.kernel_conv(kernel)
        
        return kernel
    
    
class MFLayer(nn.Module):
    def __init__(self, inc, c):
        super(MFLayer, self).__init__()
        self.rgb_conv = ConvReLU(inc, c, 3, 1, 1)
        self.hsv_conv = ConvReLU(inc, c, 3, 1, 1)
        self.lab_conv = ConvReLU(inc, c, 3, 1, 1)
        
    def forward(self, x_rgb, x_hsv, x_lab):
        x_rgb = self.rgb_conv(x_rgb)
        x_hsv = self.hsv_conv(x_hsv)
        x_lab = self.lab_conv(x_lab)
        
        return x_rgb, x_hsv, x_lab
    
    
class OutLayer(nn.Module):
    def __init__(self, c):
        super(OutLayer, self).__init__()
        self.conv1 = nn.Conv2d(c, 3, 3, 1, 1, padding_mode='reflect')
        
    def forward(self, x):
        x = self.conv1(x)
        return (torch.tanh(x) + 1) / 2    
    
    
#Encoder design    
class MFEncoder(nn.Module):     
    def __init__(self, c):
        super(MFEncoder, self).__init__()
        self.mf_conv = MFBlock(c)
        self.rgb_conv = ConvReLUBlock(c, c)
        self.hsv_conv = ConvReLUBlock(c, c)
        self.lab_conv = ConvReLUBlock(c, c)
        
    def forward(self, x_rgb, x_hsv, x_lab):
        x_agg = self.mf_conv(x_rgb, x_hsv, x_lab) 

        x_rgb = x_rgb + self.rgb_conv(x_agg)
        x_hsv = x_hsv + self.hsv_conv(x_hsv)
        x_lab = x_lab + self.lab_conv(x_lab)
        
        return x_rgb, x_hsv, x_lab    

    
class MFEncoder_single(nn.Module):     
    def __init__(self, c):
        super(MFEncoder_single, self).__init__()
        self.mf_conv = MFBlock(c)
        self.rgb_conv = ConvReLUBlock(c, c)
        
    def forward(self, x_rgb, x_hsv, x_lab):
        x_agg = self.mf_conv(x_rgb, x_hsv, x_lab) 
        x_rgb = x_rgb + self.rgb_conv(x_agg)

        return x_rgb
    
    
class MFPlainEncoder(nn.Module):     
    def __init__(self, c):
        super(MFPlainEncoder, self).__init__()
        
        self.mf_conv  = nn.Conv2d(c*3, c, 1)
        
        self.rgb_conv = ConvReLUBlock(c, c)
        self.hsv_conv = ConvReLUBlock(c, c)
        self.lab_conv = ConvReLUBlock(c, c)
        # self.outconv = depthConvReLU(outc*3, outc, True)
        
    def forward(self, x_rgb, x_hsv, x_lab):
        agg_fea = self.mf_conv(torch.cat([x_rgb, x_hsv, x_lab], dim=1))     
        
        x_rgb = x_rgb + self.rgb_conv(agg_fea)
        x_hsv = x_hsv + self.hsv_conv(x_hsv)
        x_lab = x_lab + self.lab_conv(x_lab)
        
        return x_rgb, x_hsv, x_lab

    
class MFPlainEncoder_single(nn.Module):
    def __init__(self, c):
        super(MFPlainEncoder_single, self).__init__()
        self.mf_conv  = nn.Conv2d(c*3, c, 1)
        self.rgb_conv = ConvReLUBlock(c, c)
        
    def forward(self, x_rgb, x_hsv, x_lab):
        agg_fea = self.mf_conv(torch.cat([x_rgb, x_hsv, x_lab], dim=1))            
        x_rgb = x_rgb + self.rgb_conv(agg_fea)
        
        return x_rgb
    
    
class PlainEncoder(nn.Module):     
    def __init__(self, c):
        super(PlainEncoder, self).__init__()
        
        self.rgb_conv = ConvReLUBlock(c, c)
        # self.outconv = depthConvReLU(outc*3, outc, True)
        
    def forward(self, x_rgb):
        x_rgb = x_rgb + self.rgb_conv(x_rgb)
        return x_rgb
    

#Decoder design
class MFDyDecoder(nn.Module):
    def __init__(self, c):
        super(MFDyDecoder, self).__init__()
         
        self.skip_conv  = MFBlock(c)
        self.fuse_conv  = ConvReLUBlock(c*2, c) 
        
        self.rgb_kernel = DyBlock(c)
        self.hsv_kernel = DyBlock(c)
        self.lab_kernel = DyBlock(c)
        
        self.fuse_kernel = nn.Conv2d(c*9, c*9, 1)
        
    def forward(self, x, x_rgb, x_hsv, x_lab):
        skip_feature = self.skip_conv(x_rgb, x_hsv, x_lab)
        out_feature = self.fuse_conv(torch.cat([x, skip_feature], dim=1))
        
        rgb_kernel = self.rgb_kernel(out_feature, x_rgb)        
        hsv_kernel = self.hsv_kernel(out_feature, x_hsv)        
        lab_kernel = self.lab_kernel(out_feature, x_lab)
         
        kernel = self.fuse_kernel(torch.cat([rgb_kernel, hsv_kernel, lab_kernel], dim=1))

        return out_feature, kernel
    
    
class DyDecoder(nn.Module):
    def __init__(self, c):
        super(DyDecoder, self).__init__()
         
        self.skip_conv  = ConvReLU(c*3, c, 1)
        self.fuse_conv  = ConvReLUBlock(c*2, c) 
        
        self.rgb_kernel = DyBlock(c)
        self.hsv_kernel = DyBlock(c)
        self.lab_kernel = DyBlock(c)
        
        self.fuse_kernel = nn.Conv2d(c*9, c*9, 1)
        
    def forward(self, x, x_rgb, x_hsv, x_lab):
        skip_feature = self.skip_conv(torch.cat([x_rgb, x_hsv, x_lab], dim=1))
        out_feature = self.fuse_conv(torch.cat([x, skip_feature], dim=1))
        
        rgb_kernel = self.rgb_kernel(out_feature, x_rgb)        
        hsv_kernel = self.hsv_kernel(out_feature, x_hsv)        
        lab_kernel = self.lab_kernel(out_feature, x_lab)
         
        kernel = self.fuse_kernel(torch.cat([rgb_kernel, hsv_kernel, lab_kernel], dim=1))

        return out_feature, kernel

    
class MFDecoder(nn.Module):
    def __init__(self, c):
        super(MFDecoder, self).__init__()
         
        self.skip_conv  = MFBlock(c)
        self.fuse_conv  = ConvReLUBlock(c*2, c) 
        
    def forward(self, x, x_rgb, x_hsv, x_lab):
        skip_feature = self.skip_conv(x_rgb, x_hsv, x_lab)
        out_feature = self.fuse_conv(torch.cat([x, skip_feature], dim=1))

        return out_feature    
    
    
class MFPlainDecoder(nn.Module):
    def __init__(self, c):
        super(MFPlainDecoder, self).__init__()
        '''
        upsample  
        '''   
        self.skip_conv = ConvReLU(c*3, c, 1)
        self.fuse_conv  = ConvReLUBlock(c*2, c) 
        
    def forward(self, x, x_rgb, x_hsv, x_lab):
        skip_feature = self.skip_conv(torch.cat([x_rgb, x_hsv, x_lab], dim=1))
        out_feature = self.fuse_conv(torch.cat([x, skip_feature], dim=1))

        return out_feature

    
class PlainDecoder(nn.Module):
    def __init__(self, c):
        super(PlainDecoder, self).__init__()
        '''
        upsample  
        '''   
        self.skip_conv = ConvReLU(c, c, 1)
        self.fuse_conv  = ConvReLUBlock(c*2, c) 
        
    def forward(self, x, x_rgb):
        skip_feature = self.skip_conv(x_rgb)
        out_feature = self.fuse_conv(torch.cat([x, skip_feature], dim=1))

        return out_feature    
