
class MFAtt(nn.Module):
    def __init__(self, inc, outc, num_heads):
        super(MFAtt, self).__init__()
        '''
        upsample  
        '''
        self.num_heads = num_heads
        self.norm_rgb = LayerNorm2d(inc)
        self.norm_hsv = LayerNorm2d(inc)
        self.norm_lab = LayerNorm2d(inc)
        
        self.depth_rgb = depthConv(inc, outc, False)
        self.depth_hsv = depthConv(inc, outc, False)
        self.depth_lab = depthConv(inc, outc, False)
        
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out = nn.Conv2d(outc, outc, 1)
        

    def forward(self, x_rgb, x_hsv, x_lab):
        b, c, h, w = x_rgb.shape
        rgb_inf = self.depth_rgb(self.norm_rgb(x_rgb))
        hsv_inf = self.depth_hsv(self.norm_rgb(x_hsv))
        lab_inf = self.depth_lab(self.norm_rgb(x_lab))

        rgb_inf = rearrange(rgb_inf, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        hsv_inf = rearrange(hsv_inf, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        lab_inf = rearrange(lab_inf, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        
        # print(rgb_inf.shape, hsv_inf.shape, lab_inf.shape)
        
        hsv_inf = F.normalize(hsv_inf, dim=-1)
        lab_inf = F.normalize(lab_inf, dim=-1)
        
        attn = (hsv_inf @ lab_inf.transpose(-2, -1)) 
        # print('aaa:' , attn.shape)
        # print('bbb', self.temperature.shape)
        #
        attn = attn.softmax(dim=-1)
        
        # print(attn.shape, rgb_inf.shape)
        
        out1 = (attn @ rgb_inf)
        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out1 = self.project_out(out1)
        
        return out1
    
    
class MFFeedforward(nn.Module):
    def __init__(self, inc, outc):
        super(MFFeedforward, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inc, outc, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(outc, outc, 3, 1, 1,
                      bias=False, groups=outc),
            nn.GELU(),
            nn.Conv2d(outc, outc, 1, 1, bias=False),
        )
        
    def forward(self, x):
        return self.net(x)

class MFAggEncoder(nn.Module):
    def __init__(self, inc, outc, num_heads):
        super(MFAggEncoder, self).__init__()
        self.mfatt = MFAtt(inc, outc, num_heads)
        
        self.norm_rgb = LayerNorm2d(outc)
        self.norm_hsv = LayerNorm2d(outc)
        self.norm_lab = LayerNorm2d(outc)
        
        self.mfff_rgb = MFFeedforward(outc, outc)
        self.mfff_hsv = MFFeedforward(outc, outc)
        self.mfff_lab = MFFeedforward(outc, outc)
        
    
    def forward(self, x_rgb, x_hsv, x_lab):
        x_rgb = x_rgb + self.mfatt(x_rgb, x_hsv, x_lab)
        
        x_rgb = x_rgb + self.mfff_rgb(self.norm_rgb(x_rgb))
        x_hsv = x_hsv + self.mfff_hsv(self.norm_hsv(x_hsv))
        x_lab = x_lab + self.mfff_hsv(self.norm_lab(x_lab))
        
        return x_rgb, x_hsv, x_lab
    
    
class MFAggEncoder_single(nn.Module):
    def __init__(self, inc, outc, num_heads):
        super(MFAggEncoder_single, self).__init__()
        self.mfagg = MFAtt(inc, outc, num_heads)
        self.norm_rgb = LayerNorm2d(outc)
        self.mfff_rgb = MFFeedforward(outc, outc)
         
    def forward(self, x_rgb, x_hsv, x_lab):
        x_rgb = x_rgb + self.mfagg(x_rgb, x_hsv, x_lab)
        x_rgb = x_rgb + self.mfff_rgb(self.norm_rgb(x_rgb))
        
        return x_rgb     
    

'''   
class MFDyConv(nn.Module):
    def __init__(self, c):
        super(MFDyConv, self).__init__()
        self.norm = LayerNorm2d(c)
        
        self.horDyConv1 = depthConv(c, c*3, True)
        self.horDyConv2 = depthConv(c, c*3, True)
        self.horDyConv3 = depthConv(c, c*3, True)
        
        self.verDyConv1 = depthConv(c, c*3, True)
        self.verDyConv2 = depthConv(c, c*3, True)
        self.verDyConv3 = depthConv(c, c*3, True)

        # self.dynamicConv = depthConv(inc*3, inc*3, True)
        
    def forward(self, x):
        x = self.norm(x)
        hor_1 = self.verDyConv1(x) 
        hor_2 = self.verDyConv2(x) 
        hor_3 = self.verDyConv3(x)
        
        ver_1 = self.verDyConv1(x) 
        ver_2 = self.verDyConv2(x) 
        ver_3 = self.verDyConv3(x) 
        
        b, c, h, w = hor_1.shape
         
        hor_kernel = torch.cat([hor_1, hor_2, hor_3], dim=1)
        hor_kernel = rearrange(hor_kernel, 'b (c k1 k2) h w -> b c k1 k2 h w', k1=3, k2=3)
        
        ver_kernel = torch.cat([ver_1, ver_2, ver_3], dim=1)
        ver_kernel = rearrange(ver_kernel, 'b (c k1 k2) h w -> b c k1 k2 h w', k1=3, k2=3)
        ver_kernel = rearrange(ver_kernel, 'b c k1 k2 h w -> b c k2 k1 h w', k1=3, k2=3)
        
        
        return dynamic_kernel
'''        