import torch
import torch.nn as nn
from network.kpn_network import KernelConv
import torch.nn.functional as F
from einops import rearrange
from network.MFDyNet_utils import *

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
    
    
class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]

class MFDyNet(BaseNetwork):
    def __init__(self, c=32, latent_blocks=2, init_weights=True):
        super(MFDyNet, self).__init__()     
        self.input_conv = MFLayer(3, c)
        
        self.S1_en = MFEncoder(c)       
        self.S2_en = MFEncoder(c*2)          
        self.S3_en = MFEncoder(c*4)
        
        self.down1 = Downsample3(c)
        self.down2 = Downsample3(c*2)
        self.down3 = Downsample3(c*4)
        
        blocks = []
        for _ in range(latent_blocks):
            block = MFEncoder(c*8)
            blocks.append(block)
        self.latent_layer = nn.ModuleList(blocks)
        self.latent_out_layer = MFEncoder_single(c*8)
        
        self.up3 = Upsample(c*8)
        self.up2 = Upsample(c*4)
        self.up1 = Upsample(c*2)
        
        self.S1_de = MFDyDecoder(c)      
        self.S2_de = MFDyDecoder(c*2)        
        self.S3_de = MFDyDecoder(c*4)    
        
        self.S1_out = nn.Conv2d(c, 3, 3, 1, 1, padding_mode='reflect')
        self.kernel_op = KernelConv(kernel_size=[3], sep_conv=False, core_bias=False)
        
        if init_weights is True:
            #print(init_weights)
            self.init_weights()    
            
    def __str__(self):
        return 'Multi Feature Agg Multi Feature Dynamic Network'    

        
    def forward(self, x_rgb, x_hsv, x_lab):
        #encoder
        s1_x_rgb, s1_x_hsv, s1_x_lab = self.input_conv(x_rgb, x_hsv, x_lab)
        
        #32 * H * W, ecndoer1
        s1_x_rgb, s1_x_hsv, s1_x_lab = self.S1_en(s1_x_rgb, s1_x_hsv, s1_x_lab)
        
        #64 * H/2 * W/2, encoder2
        s2_x_rgb, s2_x_hsv, s2_x_lab = self.down1(s1_x_rgb, s1_x_hsv, s1_x_lab)
        s2_x_rgb, s2_x_hsv, s2_x_lab = self.S2_en(s2_x_rgb, s2_x_hsv, s2_x_lab)
        
        #128 * H/4 * W/4, encoder3
        s3_x_rgb, s3_x_hsv, s3_x_lab = self.down2(s2_x_rgb, s2_x_hsv, s2_x_lab)
        s3_x_rgb, s3_x_hsv, s3_x_lab = self.S3_en(s3_x_rgb, s3_x_hsv, s3_x_lab)
        
        #256 * H/8 * W/8, midecoder
        s4_x_rgb, s4_x_hsv, s4_x_lab = self.down3(s3_x_rgb, s3_x_hsv, s3_x_lab)
        for layer in self.latent_layer: 
            s4_x_rgb, s4_x_hsv, s4_x_lab = layer(s4_x_rgb, s4_x_hsv, s4_x_lab) # 256*64*64
        s4_agg = self.latent_out_layer(s4_x_rgb, s4_x_hsv, s4_x_lab)    
        
        #decoder
        s3_agg = self.up3(s4_agg)
        s3_agg, s3_kernel = self.S3_de(s3_agg, s3_x_rgb, s3_x_hsv, s3_x_lab)
        s3_agg = self.kernel_op(s3_agg, s3_kernel)
        
        s2_agg = self.up2(s3_agg)
        s2_agg, s2_kernel = self.S2_de(s2_agg, s2_x_rgb, s2_x_hsv, s2_x_lab)
        s2_agg = self.kernel_op(s2_agg, s2_kernel)
                            
        s1_agg = self.up1(s2_agg)
        s1_agg, s1_kernel = self.S1_de(s1_agg, s1_x_rgb, s1_x_hsv, s1_x_lab) 
        s1_agg = self.kernel_op(s1_agg, s1_kernel)
        
        s1_out = self.S1_out(s1_agg)
        
        out = (torch.tanh(s1_out) + 1) / 2

        return out


class MFNet(BaseNetwork):
    def __init__(self, c=32, latent_blocks=2, init_weights=True):
        super(MFNet, self).__init__()     
        self.input_conv = MFLayer(3, c)
        
        self.S1_en = MFEncoder(c)       
        self.S2_en = MFEncoder(c*2)          
        self.S3_en = MFEncoder(c*4)
        
        self.down1 = Downsample3(c)
        self.down2 = Downsample3(c*2)
        self.down3 = Downsample3(c*4)
        
        blocks = []
        for _ in range(latent_blocks):
            block = MFEncoder(c*8)
            blocks.append(block)
        self.latent_layer = nn.ModuleList(blocks)
        self.latent_out_layer = MFEncoder_single(c*8)
        
        self.up3 = Upsample(c*8)
        self.up2 = Upsample(c*4)
        self.up1 = Upsample(c*2)
        
        self.S1_de = MFDecoder(c)      
        self.S2_de = MFDecoder(c*2)        
        self.S3_de = MFDecoder(c*4)    
        
        self.S1_out = nn.Conv2d(c, 3, 3, 1, 1, padding_mode='reflect')
        self.kernel_op = KernelConv(kernel_size=[3], sep_conv=False, core_bias=False)
        
        if init_weights is True:
            #print(init_weights)
            self.init_weights()    
            
    def __str__(self):
        return 'Multi Feature Agg Network'    

        
    def forward(self, x_rgb, x_hsv, x_lab):
        #encoder
        s1_x_rgb, s1_x_hsv, s1_x_lab = self.input_conv(x_rgb, x_hsv, x_lab)
        
        #32 * H * W, ecndoer1
        s1_x_rgb, s1_x_hsv, s1_x_lab = self.S1_en(s1_x_rgb, s1_x_hsv, s1_x_lab)
        
        #64 * H/2 * W/2, encoder2
        s2_x_rgb, s2_x_hsv, s2_x_lab = self.down1(s1_x_rgb, s1_x_hsv, s1_x_lab)
        s2_x_rgb, s2_x_hsv, s2_x_lab = self.S2_en(s2_x_rgb, s2_x_hsv, s2_x_lab)
        
        #128 * H/4 * W/4, encoder3
        s3_x_rgb, s3_x_hsv, s3_x_lab = self.down2(s2_x_rgb, s2_x_hsv, s2_x_lab)
        s3_x_rgb, s3_x_hsv, s3_x_lab = self.S3_en(s3_x_rgb, s3_x_hsv, s3_x_lab)
        
        #256 * H/8 * W/8, midecoder
        s4_x_rgb, s4_x_hsv, s4_x_lab = self.down3(s3_x_rgb, s3_x_hsv, s3_x_lab)
        for layer in self.latent_layer: 
            s4_x_rgb, s4_x_hsv, s4_x_lab = layer(s4_x_rgb, s4_x_hsv, s4_x_lab) # 256*64*64
        s4_agg = self.latent_out_layer(s4_x_rgb, s4_x_hsv, s4_x_lab)    
        
        #decoder
        s3_agg = self.up3(s4_agg)
        s3_agg = self.S3_de(s3_agg, s3_x_rgb, s3_x_hsv, s3_x_lab)
        
        s2_agg = self.up2(s3_agg)
        s2_agg = self.S2_de(s2_agg, s2_x_rgb, s2_x_hsv, s2_x_lab)
                            
        s1_agg = self.up1(s2_agg)
        s1_agg = self.S1_de(s1_agg, s1_x_rgb, s1_x_hsv, s1_x_lab) 
        
        s1_out = self.S1_out(s1_agg)
        
        out = (torch.tanh(s1_out) + 1) / 2

        return out
        
        
class DyNet(BaseNetwork):
    def __init__(self, c=32, latent_blocks=2, init_weights=True):
        super(DyNet, self).__init__()     
        self.input_conv = MFLayer(3, c)
        
        self.S1_en = MFPlainEncoder(c)       
        self.S2_en = MFPlainEncoder(c*2)          
        self.S3_en = MFPlainEncoder(c*4)
        
        self.down1 = Downsample3(c)
        self.down2 = Downsample3(c*2)
        self.down3 = Downsample3(c*4)
        
        blocks = []
        for _ in range(latent_blocks):
            block = MFPlainEncoder(c*8)
            blocks.append(block)
        self.latent_layer = nn.ModuleList(blocks)
        self.latent_out_layer = MFPlainEncoder_single(c*8)
        
        self.up3 = Upsample(c*8)
        self.up2 = Upsample(c*4)
        self.up1 = Upsample(c*2)
        
        self.S1_de = DyDecoder(c)      
        self.S2_de = DyDecoder(c*2)        
        self.S3_de = DyDecoder(c*4)    
        
        self.S1_out = nn.Conv2d(c, 3, 3, 1, 1, padding_mode='reflect')
        self.kernel_op = KernelConv(kernel_size=[3], sep_conv=False, core_bias=False)
        
        if init_weights is True:
            #print(init_weights)
            self.init_weights()    
            
    def __str__(self):
        return 'Multi Feature Dynamic Network'    

        
    def forward(self, x_rgb, x_hsv, x_lab):
        #encoder
        s1_x_rgb, s1_x_hsv, s1_x_lab = self.input_conv(x_rgb, x_hsv, x_lab)
        
        #32 * H * W, ecndoer1
        s1_x_rgb, s1_x_hsv, s1_x_lab = self.S1_en(s1_x_rgb, s1_x_hsv, s1_x_lab)
        
        #64 * H/2 * W/2, encoder2
        s2_x_rgb, s2_x_hsv, s2_x_lab = self.down1(s1_x_rgb, s1_x_hsv, s1_x_lab)
        s2_x_rgb, s2_x_hsv, s2_x_lab = self.S2_en(s2_x_rgb, s2_x_hsv, s2_x_lab)
        
        #128 * H/4 * W/4, encoder3
        s3_x_rgb, s3_x_hsv, s3_x_lab = self.down2(s2_x_rgb, s2_x_hsv, s2_x_lab)
        s3_x_rgb, s3_x_hsv, s3_x_lab = self.S3_en(s3_x_rgb, s3_x_hsv, s3_x_lab)
        
        #256 * H/8 * W/8, midecoder
        s4_x_rgb, s4_x_hsv, s4_x_lab = self.down3(s3_x_rgb, s3_x_hsv, s3_x_lab)
        for layer in self.latent_layer: 
            s4_x_rgb, s4_x_hsv, s4_x_lab = layer(s4_x_rgb, s4_x_hsv, s4_x_lab) # 256*64*64
        s4_agg = self.latent_out_layer(s4_x_rgb, s4_x_hsv, s4_x_lab)    
        
        #decoder
        s3_agg = self.up3(s4_agg)
        s3_agg, s3_kernel = self.S3_de(s3_agg, s3_x_rgb, s3_x_hsv, s3_x_lab)
        s3_agg = self.kernel_op(s3_agg, s3_kernel)
        
        s2_agg = self.up2(s3_agg)
        s2_agg, s2_kernel = self.S2_de(s2_agg, s2_x_rgb, s2_x_hsv, s2_x_lab)
        s2_agg = self.kernel_op(s2_agg, s2_kernel)
                             
        s1_agg = self.up1(s2_agg)
        s1_agg, s1_kernel = self.S1_de(s1_agg, s1_x_rgb, s1_x_hsv, s1_x_lab) 
        s1_agg = self.kernel_op(s1_agg, s1_kernel)
        
        s1_out = self.S1_out(s1_agg)
        
        out = (torch.tanh(s1_out) + 1) / 2

        return out
    

class Baseline3Net(BaseNetwork):
    def __init__(self, c=32, latent_blocks=2, init_weights=True):
        super(Baseline3Net, self).__init__()     
        self.input_conv = MFLayer(3, c)
        
        self.S1_en = MFPlainEncoder(c)       
        self.S2_en = MFPlainEncoder(c*2)          
        self.S3_en = MFPlainEncoder(c*4)
        
        self.down1 = Downsample3(c)
        self.down2 = Downsample3(c*2)
        self.down3 = Downsample3(c*4)
        
        blocks = []
        for _ in range(latent_blocks):
            block = MFPlainEncoder(c*8)
            blocks.append(block)
        self.latent_layer = nn.ModuleList(blocks)
        self.latent_out_layer = MFPlainEncoder_single(c*8)
        
        self.up3 = Upsample(c*8)
        self.up2 = Upsample(c*4)
        self.up1 = Upsample(c*2)
        
        self.S1_de = MFPlainDecoder(c)      
        self.S2_de = MFPlainDecoder(c*2)        
        self.S3_de = MFPlainDecoder(c*4)    
        
        self.S1_out = nn.Conv2d(c, 3, 3, 1, 1, padding_mode='reflect')
        self.kernel_op = KernelConv(kernel_size=[3], sep_conv=False, core_bias=False)
        
        if init_weights is True:
            #print(init_weights)
            self.init_weights()    
            
    def __str__(self):
        return '3 Color Network'    

        
    def forward(self, x_rgb, x_hsv, x_lab):
        #encoder
        s1_x_rgb, s1_x_hsv, s1_x_lab = self.input_conv(x_rgb, x_hsv, x_lab)
        
        #32 * H * W, ecndoer1
        s1_x_rgb, s1_x_hsv, s1_x_lab = self.S1_en(s1_x_rgb, s1_x_hsv, s1_x_lab)
        
        #64 * H/2 * W/2, encoder2
        s2_x_rgb, s2_x_hsv, s2_x_lab = self.down1(s1_x_rgb, s1_x_hsv, s1_x_lab)
        s2_x_rgb, s2_x_hsv, s2_x_lab = self.S2_en(s2_x_rgb, s2_x_hsv, s2_x_lab)
        
        #128 * H/4 * W/4, encoder3
        s3_x_rgb, s3_x_hsv, s3_x_lab = self.down2(s2_x_rgb, s2_x_hsv, s2_x_lab)
        s3_x_rgb, s3_x_hsv, s3_x_lab = self.S3_en(s3_x_rgb, s3_x_hsv, s3_x_lab)
        
        #256 * H/8 * W/8, midecoder
        s4_x_rgb, s4_x_hsv, s4_x_lab = self.down3(s3_x_rgb, s3_x_hsv, s3_x_lab)
        for layer in self.latent_layer: 
            s4_x_rgb, s4_x_hsv, s4_x_lab = layer(s4_x_rgb, s4_x_hsv, s4_x_lab) # 256*64*64
        s4_agg = self.latent_out_layer(s4_x_rgb, s4_x_hsv, s4_x_lab)    
        
        #decoder
        s3_agg = self.up3(s4_agg)
        s3_agg = self.S3_de(s3_agg, s3_x_rgb, s3_x_hsv, s3_x_lab)
        
        s2_agg = self.up2(s3_agg)
        s2_agg = self.S2_de(s2_agg, s2_x_rgb, s2_x_hsv, s2_x_lab)
                            
        s1_agg = self.up1(s2_agg)
        s1_agg = self.S1_de(s1_agg, s1_x_rgb, s1_x_hsv, s1_x_lab) 
        
        s1_out = self.S1_out(s1_agg)
        
        out = (torch.tanh(s1_out) + 1) / 2

        return out
        
class Baseline1Net(BaseNetwork):
    def __init__(self, c=32, latent_blocks=2, init_weights=True):
        super(Baseline1Net, self).__init__()     
        self.input_conv = ConvReLU(3, c, 3, 1, 1)
        
        self.S1_en = PlainEncoder(c)       
        self.S2_en = PlainEncoder(c*2)          
        self.S3_en = PlainEncoder(c*4)
        
        self.down1 = Downsample(c)
        self.down2 = Downsample(c*2)
        self.down3 = Downsample(c*4)
        
        blocks = []
        for _ in range(latent_blocks):
            block = PlainEncoder(c*8)
            blocks.append(block)
        self.latent_layer = nn.ModuleList(blocks)
        self.latent_out_layer = PlainEncoder(c*8)
        
        self.up3 = Upsample(c*8)
        self.up2 = Upsample(c*4)
        self.up1 = Upsample(c*2)
        
        self.S1_de = PlainDecoder(c)      
        self.S2_de = PlainDecoder(c*2)        
        self.S3_de = PlainDecoder(c*4)    
        
        self.S1_out = nn.Conv2d(c, 3, 3, 1, 1, padding_mode='reflect')
        self.kernel_op = KernelConv(kernel_size=[3], sep_conv=False, core_bias=False)
        
        if init_weights is True:
            #print(init_weights)
            self.init_weights()    
            
    def __str__(self):
        return '1 Color Network'    

        
    def forward(self, x_rgb, hsv_holder, lab_holder):
        #encoder
        s1_x_rgb = self.input_conv(x_rgb)
        
        #32 * H * W, ecndoer1
        s1_x_rgb = self.S1_en(s1_x_rgb)
        
        #64 * H/2 * W/2, encoder2
        s2_x_rgb = self.down1(s1_x_rgb)
        s2_x_rgb = self.S2_en(s2_x_rgb)
        
        #128 * H/4 * W/4, encoder3
        s3_x_rgb = self.down2(s2_x_rgb)
        s3_x_rgb = self.S3_en(s3_x_rgb)
        
        #256 * H/8 * W/8, midecoder
        s4_x_rgb = self.down3(s3_x_rgb)
        for layer in self.latent_layer: 
            s4_x_rgb = layer(s4_x_rgb) # 256*64*64
        s4_agg = self.latent_out_layer(s4_x_rgb)    
        # s4_agg = s4_x_rgb
            
        #decoder
        s3_agg = self.up3(s4_agg)
        s3_agg = self.S3_de(s3_agg, s3_x_rgb)
        
        s2_agg = self.up2(s3_agg)
        s2_agg = self.S2_de(s2_agg, s2_x_rgb)
                            
        s1_agg = self.up1(s2_agg)
        s1_agg = self.S1_de(s1_agg, s1_x_rgb) 
        
        s1_out = self.S1_out(s1_agg)
        
        out = (torch.tanh(s1_out) + 1) / 2

        return out
        
# class MFDyNet(BaseNetwork):
#     def __init__(self, c=32, latent_blocks=4, init_weights=True):
#         super(MFDyNet, self).__init__()     
#         self.input_conv = MFLayer(3, c)
        
#         self.S1_en = MFEncoder(c)       
#         self.S2_en = MFEncoder(c*2)          
#         self.S3_en = MFEncoder(c*4)
        
#         self.down1 = Downsample3(c)
#         self.down2 = Downsample3(c*2)
#         self.down3 = Downsample3(c*4)
        
#         blocks = []
#         for _ in range(latent_blocks):
#             block = MFEncoder(c*8)
#             blocks.append(block)
#         self.latent_layer = nn.ModuleList(blocks)
#         self.latent_out_layer = MFEncoder_single(c*8)
        
#         self.up3 = Upsample(c*8)
#         self.up2 = Upsample(c*4)
#         self.up1 = Upsample(c*2)
        
#         self.S1_de = MFDyDecoder(c)      
#         self.S2_de = MFDyDecoder(c*2)        
#         self.S3_de = MFDyDecoder(c*4)    
        
#         self.S1_out = nn.Conv2d(c, 3, 3, 1, 1, padding_mode='reflect')
#         self.kernel_op = KernelConv(kernel_size=[3], sep_conv=False, core_bias=False)
        
#         if init_weights is True:
#             #print(init_weights)
#             self.init_weights()    
            
#     def __str__(self):
#         return 'Multi Feature Dynamic Network'    

        
#     def forward(self, x_rgb, x_hsv, x_lab):
#         #encoder
#         s1_x_rgb, s1_x_hsv, s1_x_lab = self.input_conv(x_rgb, x_hsv, x_lab)
        
#         #32 * H * W, ecndoer1
#         s1_x_rgb, s1_x_hsv, s1_x_lab = self.S1_en(s1_x_rgb, s1_x_hsv, s1_x_lab)
        
#         #64 * H/2 * W/2, encoder2
#         s2_x_rgb, s2_x_hsv, s2_x_lab = self.down1(s1_x_rgb, s1_x_hsv, s1_x_lab)
#         s2_x_rgb, s2_x_hsv, s2_x_lab = self.S2_en(s2_x_rgb, s2_x_hsv, s2_x_lab)
        
#         #128 * H/4 * W/4, encoder3
#         s3_x_rgb, s3_x_hsv, s3_x_lab = self.down2(s2_x_rgb, s2_x_hsv, s2_x_lab)
#         s3_x_rgb, s3_x_hsv, s3_x_lab = self.S3_en(s3_x_rgb, s3_x_hsv, s3_x_lab)
        
#         #256 * H/8 * W/8, midecoder
#         s4_x_rgb, s4_x_hsv, s4_x_lab = self.down3(s3_x_rgb, s3_x_hsv, s3_x_lab)
#         for layer in self.latent_layer: 
#             s4_x_rgb, s4_x_hsv, s4_x_lab = layer(s4_x_rgb, s4_x_hsv, s4_x_lab) # 256*64*64
#         s4_agg = self.latent_out_layer(s4_x_rgb, s4_x_hsv, s4_x_lab)
            
#         #decoder
#         s3_agg = self.up3(s4_agg)
#         s3_agg, s3_kernel = self.S3_de(s3_agg, s3_x_rgb, s3_x_hsv, s3_x_lab)
#         s3_agg = self.kernel_op(s3_agg, s3_kernel)
        
#         s2_agg = self.up2(s3_agg)
#         s2_agg, s2_kernel = self.S2_de(s2_agg, s2_x_rgb, s2_x_hsv, s2_x_lab)
#         s2_agg = self.kernel_op(s2_agg, s2_kernel)
                            
#         s1_agg = self.up1(s2_agg)
#         s1_agg, s1_kernel = self.S1_de(s1_agg, s1_x_rgb, s1_x_hsv, s1_x_lab) 
#         s1_agg = self.kernel_op(s1_agg, s1_kernel)
        
#         s1_out = self.S1_out(s1_agg)
        
#         out = (torch.tanh(s1_out) + 1) / 2

#         return out

    
    
class p2p(BaseNetwork):
    def __init__(self, c=64, residual_blocks=2, init_weights=True):
        super(p2p, self).__init__()     
        self.input_conv = MFLayer(3, c)
        
        self.S1_en = MFPlainEncoder(c, c, 1)       
        self.S2_en = MFPlainEncoder(c*2, c*2, 2)          
        self.S3_en = MFPlainEncoder(c*4, c*4, 4)
        self.S4_en = MFPlainEncoder_single(c*8, c*8, 8)
        
        self.down1 = Downsample3(c)
        self.down2 = Downsample3(c*2)
        self.down3 = Downsample3(c*4)
        
        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(c*8, 2)
            blocks.append(block)

        self.midcoder = nn.ModuleList(blocks)
        
        self.up3 = Upsample(c*8)
        self.up2 = Upsample(c*4)
        self.up1 = Upsample(c*2)
        
        self.S1_de = MFDecoder(c, c)      
        self.S2_de = MFDecoder(c*2, c*2)        
        self.S3_de = MFDecoder(c*4, c*4)    
        
        self.S1_out = nn.Sequential(nn.ReflectionPad2d(3),
                                        nn.Conv2d(in_channels=c, out_channels=3, kernel_size=7, padding=0),)
        
        if init_weights is True:
            #print(init_weights)
            self.init_weights()    
            
    def __str__(self):
        return 'Baseline'    

        
    def forward(self, x_rgb, x_hsv, x_lab):
        #encoder

        s1_x_rgb, s1_x_hsv, s1_x_lab = self.input_conv(x_rgb, x_hsv, x_lab)
        
        #32 * H * W, ecndoer1
        s1_x_rgb, s1_x_hsv, s1_x_lab = self.S1_en(s1_x_rgb, s1_x_hsv, s1_x_lab)
        
        #64 * H/2 * W/2, encoder2
        s2_x_rgb, s2_x_hsv, s2_x_lab = self.down1(s1_x_rgb, s1_x_hsv, s1_x_lab)
        s2_x_rgb, s2_x_hsv, s2_x_lab = self.S2_en(s2_x_rgb, s2_x_hsv, s2_x_lab)
        
        
        #128 * H/4 * W/4, encoder3
        s3_x_rgb, s3_x_hsv, s3_x_lab = self.down2(s2_x_rgb, s2_x_hsv, s2_x_lab)
        s3_x_rgb, s3_x_hsv, s3_x_lab = self.S3_en(s3_x_rgb, s3_x_hsv, s3_x_lab)
        
        #256 * H/8 * W/8, encoder4
        s4_x_rgb, s4_x_hsv, s4_x_lab = self.down3(s3_x_rgb, s3_x_hsv, s3_x_lab)
        s4_x_rgb = self.S4_en(s4_x_rgb, s4_x_hsv, s4_x_lab)
        
        #midcoder
        mid = s4_x_rgb
        for layer in self.midcoder: 
            mid = layer((mid, None)) # 256*64*64

        #decoder
        s3_agg = self.up3(mid)
        s3_agg = self.S3_de(s3_agg, s3_x_rgb, s3_x_hsv, s3_x_lab)
        
        s2_agg = self.up2(s3_agg)
        s2_agg = self.S2_de(s2_agg, s2_x_rgb, s2_x_hsv, s2_x_lab)
                            
        s1_agg = self.up1(s2_agg)
        s1_agg = self.S1_de(s1_agg, s1_x_rgb, s1_x_hsv, s1_x_lab) 
        
        s1_out = self.S1_out(s1_agg)
        
        out = (torch.tanh(s1_out) + 1) / 2

        return out
