import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
import network.kpn_utils as kpn_utils
from network.kpn_network import KernelConv
from network import clf
from network.DANet_utils import DABlock, NAFBlock, MRBlock, simpleMRBlock, LayerNorm2d


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

    def forward(self, block_inputs):
        x, _ = block_inputs
        out = x + self.conv_block(x)

        return out

class pix2pix(BaseNetwork):
    def __init__(self, c=64, residual_blocks=8, init_weights=True):
        super(pix2pix, self).__init__()
        
        self.head = nn.Sequential(nn.ReflectionPad2d(3),
                                        nn.Conv2d(in_channels=3, out_channels=c, kernel_size=7, padding=0),
                                        # nn.InstanceNorm2d(c),
                                        nn.ReLU(True),)

        self.encoder0 = nn.Sequential(nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1),
                                        # nn.InstanceNorm2d(c*2),
                                        nn.ReLU(True),)

        self.encoder1 = nn.Sequential(nn.Conv2d(in_channels=c*2, out_channels=c*4, kernel_size=4, stride=2, padding=1),
                                        # nn.InstanceNorm2d(c*4),
                                        nn.ReLU(True),)

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(c*4, 2)
            blocks.append(block)

        self.midcoder = nn.ModuleList(blocks)

        self.decoder1 = nn.Sequential(nn.ConvTranspose2d(in_channels=c*4, out_channels=c*2, kernel_size=4, stride=2, padding=1),
                                        # nn.InstanceNorm2d(c*2),
                                        nn.ReLU(True),)
        self.decoder0 = nn.Sequential(nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1),
                                        # nn.InstanceNorm2d(c),
                                        nn.ReLU(True),)
        self.tail = nn.Sequential(nn.ReflectionPad2d(3),
                                        nn.Conv2d(in_channels=c, out_channels=3, kernel_size=7, padding=0),)
        
        if init_weights is True:
            #print(init_weights)
            self.init_weights()    
            
    def __str__(self):
        return 'Baseline'    

        
    def forward(self, ins, flag_clf_clean):
        x, hq_x = ins
        if flag_clf_clean is True:
            clf_x = torch.cat([x, hq_x], dim=0)
        else:
            clf_x = x
        inputs = x.clone()
        
        #encoder
        x = self.head(x)     # 64*256*256
        x = self.encoder0(x) # 128*128*128
        x = self.encoder1(x) # 256*64*64
        
        #midcoder
        for layer in self.midcoder: 
            x = layer((x, None)) # 512*32*32

        #decoder
        x = self.decoder1(x)    # 128*128*128
        x = self.decoder0(x)    # 64*256*256
        x = self.tail(x)            # 3*256*256

        out = (torch.tanh(x) + 1) / 2

        return out, None

    def save_feature(self, x, name):
        x = x.cpu().numpy()
        np.save('./result/{}'.format(name), x)

    
class pix2pixW2clf(BaseNetwork):
    def __init__(self, c=64, residual_blocks=8, init_weights=True):
        super(pix2pixW2clf, self).__init__()
        
        self.head = nn.Sequential(nn.ReflectionPad2d(3),
                                        nn.Conv2d(in_channels=3, out_channels=c, kernel_size=7, padding=0),
                                        nn.InstanceNorm2d(c),
                                        nn.ReLU(True),)

        self.encoder0 = nn.Sequential(nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1),
                                        nn.InstanceNorm2d(c*2),
                                        nn.ReLU(True),)

        self.encoder1 = nn.Sequential(nn.Conv2d(in_channels=c*2, out_channels=c*4, kernel_size=4, stride=2, padding=1),
                                        nn.InstanceNorm2d(c*4),
                                        nn.ReLU(True),)


        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(c*4, 2)
            blocks.append(block)

        self.midcoder = nn.ModuleList(blocks)

        self.decoder1 = nn.Sequential(nn.ConvTranspose2d(in_channels=c*4, out_channels=c*2, kernel_size=4, stride=2, padding=1),
                                        nn.InstanceNorm2d(c*2),
                                        nn.ReLU(True),)
        self.decoder0 = nn.Sequential(nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1),
                                        nn.InstanceNorm2d(c),
                                        nn.ReLU(True),)
        self.tail = nn.Sequential(nn.ReflectionPad2d(3),
                                        nn.Conv2d(in_channels=c, out_channels=3, kernel_size=7, padding=0),)
        
        self.resnet = clf.ResNet(64)
        self.active = nn.LogSoftmax(dim=1)

        self.kpnmlplayer = nn.Sequential(nn.Conv2d(64*8, c*4, 1),
                                         nn.ReLU(),
                                         nn.Conv2d(c*4, c*4, 1),
                                         nn.Sigmoid())
        
        self.kpnmlplayer2 = nn.Sequential(nn.Conv2d(64*8, c*4, 1),
                                 nn.ReLU(),
                                 nn.Conv2d(c*4, c, 1),
                                 nn.Sigmoid())
        
        if init_weights is True:
            #print(init_weights)
            self.init_weights()
            
    def __str__(self):
        return 'Baseline + Degradation classifier'    

        
    def forward(self, ins, flag_clf_clean):
        x, hq_x = ins
        if flag_clf_clean is True:
            clf_x = torch.cat([x, hq_x], dim=0)
        else:
            clf_x = x
        inputs = x.clone()
        
        #degradation classifier
        de_output, de_fea = self.resnet(clf_x)
        if flag_clf_clean is True:
            de_fea = de_fea[:-1]
        de_output = self.active(de_output)
        #print(de_output.shape)
        
        #degradation attention 
        kpn_activ = self.kpnmlplayer(de_fea)
        kpn_activ2 = self.kpnmlplayer2(de_fea)

        #encoder
        x = self.head(x)
        x = self.encoder0(x) # 128*128*128
        x = self.encoder1(x) # 256*64*64
        x = x * kpn_activ
        
        #midcoder
        for layer in self.midcoder: 
            x = layer((x, None)) # 256*64*64

        #decoder
        x = self.decoder1(x)    #64*256*256
        x = self.decoder0(x)    #3*256*256
        x = x * kpn_activ2
        x = self.tail(x)

        out = (torch.tanh(x) + 1) / 2

        return out, de_output

    def save_feature(self, x, name):
        x = x.cpu().numpy()
        np.save('./result/{}'.format(name), x)
        
    
class pix2pixWdfm(BaseNetwork):
    def __init__(self, c=64, residual_blocks=8, init_weights=True):
        super(pix2pixWdfm, self).__init__()
        
        self.head = nn.Sequential(nn.ReflectionPad2d(3),
                                        nn.Conv2d(in_channels=3, out_channels=c, kernel_size=7, padding=0),
                                        nn.InstanceNorm2d(c),
                                        nn.ReLU(True),)

        self.encoder0 = nn.Sequential(nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1),
                                        nn.InstanceNorm2d(c*2),
                                        nn.ReLU(True),)

        self.encoder1 = nn.Sequential(nn.Conv2d(in_channels=c*2, out_channels=c*4, kernel_size=4, stride=2, padding=1),
                                        nn.InstanceNorm2d(c*4),
                                        nn.ReLU(True),)

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(c*4, 2)
            blocks.append(block)

        self.midcoder = nn.ModuleList(blocks)

        self.decoder1 = nn.Sequential(nn.ConvTranspose2d(in_channels=c*4, out_channels=c*2, kernel_size=4, stride=2, padding=1),
                                        nn.InstanceNorm2d(c*2),
                                        nn.ReLU(True),)
        self.decoder0 = nn.Sequential(nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1),
                                        nn.InstanceNorm2d(c),
                                        nn.ReLU(True),)
        self.tail = nn.Sequential(nn.ReflectionPad2d(3),
                                        nn.Conv2d(in_channels=c, out_channels=3, kernel_size=7, padding=0),)

        self.kernel_pred = KernelConv(kernel_size=[3], sep_conv=False, core_bias=False)

        self.kpn_model = kpn_utils.create_generator()
        
        if init_weights is True:
            #print(init_weights)
            self.init_weights()  
            
    def __str__(self):
        return 'Baseline + Dynamic filter'    

        
    def forward(self, ins, flag_clf_clean):
        x, hq_x = ins
        if flag_clf_clean is True:
            clf_x = torch.cat([x, hq_x], dim=0)
        else:
            clf_x = x
        inputs = x.clone()
        
        #encoder
        x = self.head(x) # 64*256*256
        x = self.encoder0(x) # 128*128*128
        kernels, kernels_img = self.kpn_model(inputs, x)
        x = self.encoder1(x) # 256*64*64
        x = self.kernel_pred(x, kernels, white_level=1.0, rate=1)
        
        #midcoder
        for layer in self.midcoder: 
            x = layer((x, None)) # 512*32*32

        #decoder
        x = self.decoder1(x)    #64*256*256
        x = self.decoder0(x)    #3*256*256
        x = self.tail(x)

        x = self.kernel_pred(x, kernels_img, white_level=1.0, rate=1)

        out = (torch.tanh(x) + 1) / 2

        return out, None

    def save_feature(self, x, name):
        x = x.cpu().numpy()
        np.save('./result/{}'.format(name), x)

        
class pix2pixWdfmW2clf(BaseNetwork):
    def __init__(self, c=64, residual_blocks=8, init_weights=True):
        super(pix2pixWdfmW2clf, self).__init__()
        
        self.head = nn.Sequential(nn.ReflectionPad2d(3),
                                        nn.Conv2d(in_channels=3, out_channels=c, kernel_size=7, padding=0),
                                        nn.InstanceNorm2d(c),
                                        nn.ReLU(True),)

        self.encoder0 = nn.Sequential(nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1),
                                        nn.InstanceNorm2d(c*2),
                                        nn.ReLU(True),)

        self.encoder1 = nn.Sequential(nn.Conv2d(in_channels=c*2, out_channels=c*4, kernel_size=4, stride=2, padding=1),
                                        nn.InstanceNorm2d(c*4),
                                        nn.ReLU(True),)


        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(c*4, 2)
            blocks.append(block)

        self.midcoder = nn.ModuleList(blocks)

        self.decoder1 = nn.Sequential(nn.ConvTranspose2d(in_channels=c*4, out_channels=c*2, kernel_size=4, stride=2, padding=1),
                                        nn.InstanceNorm2d(c*2),
                                        nn.ReLU(True),)
        self.decoder0 = nn.Sequential(nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1),
                                        nn.InstanceNorm2d(c),
                                        nn.ReLU(True),)
        self.tail = nn.Sequential(nn.ReflectionPad2d(3),
                                        nn.Conv2d(in_channels=c, out_channels=3, kernel_size=7, padding=0),)

        self.kernel_pred = KernelConv(kernel_size=[3], sep_conv=False, core_bias=False)

        self.kpn_model = kpn_utils.create_generatorV3()
        
        self.resnet = clf.ResNet(64)
        self.active = nn.LogSoftmax(dim=1)

        self.kpnmlplayer = nn.Sequential(nn.Conv2d(64*8, c*4, 1),
                                         nn.ReLU(),
                                         nn.Conv2d(c*4, c*4, 1),
                                         nn.Sigmoid())
        
        self.kpnmlplayer2 = nn.Sequential(nn.Conv2d(64*8, c*4, 1),
                                 nn.ReLU(),
                                 nn.Conv2d(c*4, c, 1),
                                 nn.Sigmoid())
        
        if init_weights is True:
            #print(init_weights)
            self.init_weights()
            
    def __str__(self):
        return 'Baseline + Dynamic filter + Degradation classifier'    

        
    def forward(self, ins, flag_clf_clean):
        x, hq_x = ins
        if flag_clf_clean is True:
            clf_x = torch.cat([x, hq_x], dim=0)
        else:
            clf_x = x
        inputs = x.clone()
        
        #degradation classifier
        de_output, de_fea = self.resnet(clf_x)
        if flag_clf_clean is True:
            de_fea = de_fea[:-1]
        de_output = self.active(de_output)
        #print(de_output.shape)
        
        #degradation attention 
        kpn_activ = self.kpnmlplayer(de_fea)
        kpn_activ2 = self.kpnmlplayer2(de_fea)

        #encoder
        x = self.head(x)
        x = self.encoder0(x) # 128*128*128
        kernels, kernels_img = self.kpn_model(inputs, x, kpn_activ, kpn_activ2)
        x = self.encoder1(x) # 256*64*64
        x = self.kernel_pred(x, kernels, white_level=1.0, rate=1)
        
        #midcoder
        for layer in self.midcoder: 
            x = layer((x, None)) # 256*64*64

        #decoder
        x = self.decoder1(x)    #64*256*256
        x = self.decoder0(x)    #3*256*256
        x = self.tail(x)
        x = self.kernel_pred(x, kernels_img, white_level=1.0, rate=1)

        out = (torch.tanh(x) + 1) / 2

        return out, de_output

    def save_feature(self, x, name):
        x = x.cpu().numpy()
        np.save('./result/{}'.format(name), x)
        
        
class pix2pixWdfmW2clfReivision(BaseNetwork):
    def __init__(self, att_level, c=64, residual_blocks=8, init_weights=True):
        super(pix2pixWdfmW2clfReivision, self).__init__()
        
        self.head = nn.Sequential(nn.ReflectionPad2d(3),
                                        nn.Conv2d(in_channels=3, out_channels=c, kernel_size=7, padding=0),
                                        nn.InstanceNorm2d(c),
                                        nn.ReLU(True),)

        self.encoder0 = nn.Sequential(nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1),
                                        nn.InstanceNorm2d(c*2),
                                        nn.ReLU(True),)

        self.encoder1 = nn.Sequential(nn.Conv2d(in_channels=c*2, out_channels=c*4, kernel_size=4, stride=2, padding=1),
                                        nn.InstanceNorm2d(c*4),
                                        nn.ReLU(True),)


        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(c*4, 2)
            blocks.append(block)

        self.midcoder = nn.ModuleList(blocks)

        self.decoder1 = nn.Sequential(nn.ConvTranspose2d(in_channels=c*4, out_channels=c*2, kernel_size=4, stride=2, padding=1),
                                        nn.InstanceNorm2d(c*2),
                                        nn.ReLU(True),)
        self.decoder0 = nn.Sequential(nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1),
                                        nn.InstanceNorm2d(c),
                                        nn.ReLU(True),)
        self.tail = nn.Sequential(nn.ReflectionPad2d(3),
                                        nn.Conv2d(in_channels=c, out_channels=3, kernel_size=7, padding=0),)

        self.kernel_pred = KernelConv(kernel_size=[3], sep_conv=False, core_bias=False)

        self.kpn_model = kpn_utils.create_generatorRevision(att_level)
            
        self.resnet = clf.ResNet(64)
        self.active = nn.LogSoftmax(dim=1)

        self.kpnmlplayer = nn.Sequential(nn.Conv2d(64*8, c*4, 1),
                                         nn.ReLU(),
                                         nn.Conv2d(c*4, c*4, 1),
                                         nn.Sigmoid())
        
        self.kpnmlplayer2 = nn.Sequential(nn.Conv2d(64*8, c*4, 1),
                                 nn.ReLU(),
                                 nn.Conv2d(c*4, c, 1),
                                 nn.Sigmoid())
        
        if init_weights is True:
            #print(init_weights)
            self.init_weights()
            
    def __str__(self):
        return 'Baseline + Dynamic filter + Degradation classifier + one filter'    

        
    def forward(self, ins, flag_clf_clean):
        x, hq_x = ins
        if flag_clf_clean is True:
            clf_x = torch.cat([x, hq_x], dim=0)
        else:
            clf_x = x
        inputs = x.clone()
        
        #degradation classifier
        de_output, de_fea = self.resnet(clf_x)
        if flag_clf_clean is True:
            de_fea = de_fea[:-1]
        de_output = self.active(de_output)
        #print(de_output.shape)
        
        #degradation attention 
        kpn_activ = self.kpnmlplayer(de_fea)
        kpn_activ2 = self.kpnmlplayer2(de_fea)

        #encoder
        x = self.head(x)
        x = self.encoder0(x) # 128*128*128
        kernels, kernels_img = self.kpn_model(inputs, x, kpn_activ, kpn_activ2)
        x = self.encoder1(x) # 256*64*64
        x = self.kernel_pred(x, kernels, white_level=1.0, rate=1)
        
        #midcoder
        for layer in self.midcoder: 
            x = layer((x, None)) # 256*64*64

        #decoder
        x = self.decoder1(x)    #64*256*256
        x = self.decoder0(x)    #3*256*256
        x = self.tail(x)
        x = self.kernel_pred(x, kernels_img, white_level=1.0, rate=1)

        out = (torch.tanh(x) + 1) / 2

        return out, de_output

    def save_feature(self, x, name):
        x = x.cpu().numpy()
        np.save('./result/{}'.format(name), x)
        

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

        if init_weights:
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
    
        
def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class DiscriminatorV2(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(DiscriminatorV2, self).__init__()
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

        if init_weights:
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

        return outputs, conv4


class IRModel(BaseNetwork):
    def __init__(self, in_channels, out_channels, use_spectral_norm=True, init_weights=True):
        super(IRModel, self).__init__()
        
        self.convR = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
                )
        
        if init_weights:
            self.init_weights()

    def forward(self, x):

        return self.convR(x)