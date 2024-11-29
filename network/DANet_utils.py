import torch.nn as nn
import torch
from network.kpn_network import KernelConv
import torch.nn.functional as F

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

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
class AddOp(nn.Module):
    def __init__(self, c):
        super(AddOp, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c,
                               bias=True)
        #self.sg = SimpleGate()
    def forward(self, x):
        return self.conv2(self.conv1(x))
        
        
class MulOp(nn.Module):
    def __init__(self, c):
        super(MulOp, self).__init__()
        self.structure1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.structure2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c,
                               bias=True)
        
        self.color1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.color2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c,
                               bias=True)
        # self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, \
        #                        padding=0, stride=1, groups=1, bias=True)
        # self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, \
        #                        padding=1, stride=1, groups=c, bias=True)
        
        self.fuse = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
    def forward(self, x):
        x_s = self.structure2(self.structure1(x))
        x_c = self.color2(self.color1(x))
        out = self.fuse(x_s*x_c)
        return out
    
class simpleMulOp(nn.Module):
    def __init__(self, c):
        super(simpleMulOp, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, \
                               padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, \
                               padding=1, stride=1, groups=c, bias=True)
        
    def forward(self, x):
        return x*self.conv2(self.conv1(x))    

    
class ConvOp(nn.Module):
    def __init__(self, c, kernel_size=3):
        super(ConvOp, self).__init__()
        self.kernel_c = c // 4 * (kernel_size**2)
        
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, \
                               padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, \
                               padding=1, stride=1, groups=c, bias=True)
        
        self.kernel_conv = nn.Conv2d(c, self.kernel_c, 1, 1, 0)
        
        self.kernel_pred = KernelConv(kernel_size=[3], sep_conv=False, core_bias=False)
        
        
    def forward(self, x):
        kernels = self.conv2(self.conv1(x))
        kernels = self.kernel_conv(kernels)
        kernels = kernels.unsqueeze(dim=0)      
        
        #print('kernel shape before interpolate:', kernels.shape)      
        kernels = F.interpolate(input=kernels, size=(self.kernel_c * 4, x.shape[-1], x.shape[-2]), mode='nearest')
        #print('kernel shape after interpolate:', kernels.shape)
        
        kernels = kernels.squeeze(dim=0)   
        out = self.kernel_pred(x, kernels, white_level=1.0, rate=1)
        
        return out
    
class SelfAttOp(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super(SelfAttOp, self).__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        # self.dropout1 = nn.Dr`opout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        # x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        # x = self.dropout2(x)

        return y + x * self.gamma

class MRBlock(nn.Module):
    def __init__(self, c):
        super(MRBlock, self).__init__()
        self.add_op = AddOp(c)
        self.mul_op = MulOp(c)
        self.conv_op = ConvOp(c)
        self.norm = LayerNorm2d(c)
        self.fus_op = nn.Sequential(nn.Conv2d(in_channels=c*3, out_channels=c, kernel_size=1),
                                    nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=1, \
                                              padding=0, stride=1, groups=1, bias=True),
                                    SimpleGate(),
                                    nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, \
                                              padding=0, stride=1, groups=1, bias=True),
                                   )
        
    def forward(self, block_inputs):
        x, degrad_att = block_inputs
        inx = self.norm(x)
        add_x = self.add_op(inx)
        mul_x = self.mul_op(inx)
        conv_x = self.conv_op(inx)        
        out = torch.cat([add_x, mul_x, conv_x], dim=1) * degrad_att
        out = self.fus_op(out)

        return x + out
    
    
class simpleMRBlock(nn.Module):
    def __init__(self, c):
        super(simpleMRBlock, self).__init__()
        self.add_op = AddOp(c)
        self.mul_op = simpleMulOp(c)
        self.conv_op = ConvOp(c)
        self.norm = LayerNorm2d(c)
        self.fus_op = nn.Sequential(nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=1, \
                                              padding=0, stride=1, groups=1, bias=True),
                                    SimpleGate(),
                                    nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, \
                                              padding=0, stride=1, groups=1, bias=True),
                                   )
        
    def forward(self, block_inputs):
        x, degrad_att = block_inputs
        inx = self.norm(x)
        add_x = self.add_op(inx)
        mul_x = self.mul_op(inx)
        conv_x = self.conv_op(inx)        
        out = (add_x+mul_x+conv_x) * degrad_att
        out = self.fus_op(out)

        return x + out

    
class DABlock(nn.Module):
    def __init__(self, c):
        super(DABlock, self).__init__()
        self.selfatt_op = SelfAttOp(c)
        
        self.add_op = AddOp(c)
        self.mul_op = MulOp(c)
        self.conv_op = ConvOp(c)
        self.norm = LayerNorm2d(c)
        self.fus_op = nn.Sequential(nn.Conv2d(in_channels=c*3, out_channels=c, kernel_size=1),
                                    nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=1, \
                                              padding=0, stride=1, groups=1, bias=True),
                                    SimpleGate(),
                                    nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, \
                                              padding=0, stride=1, groups=1, bias=True),
                                   )
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        
    def forward(self, block_inputs):
        x, degrad_att = block_inputs

        #layer 1
        x = self.selfatt_op(x)
    
        #layer 2
        inx = self.norm(x)
        add_x = self.add_op(inx)
        mul_x = self.mul_op(inx)
        conv_x = self.conv_op(inx)        
        out = torch.cat([add_x, mul_x, conv_x], dim=1) * degrad_att
        out = self.fus_op(out)
        
        return x + out
    
class NAFBlock(nn.Module):
    def __init__(self, c):
        super(NAFBlock, self).__init__()
        self.selfatt_op = SelfAttOp(c)

    def forward(self, block_inputs):
        x, _ = block_inputs
        x = self.selfatt_op(x)
        
        return x