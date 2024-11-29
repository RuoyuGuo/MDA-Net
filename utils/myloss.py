import torch
import torch.nn as nn
import torch.nn.functional as F
import functools   

   
class GANLoss(nn.Module):
    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(GANLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss

class IRLoss(nn.Module):
    def __init__(self, pos_ratio=0.5):
        super(IRLoss, self).__init__()
        self.pos_ratio = pos_ratio
        
    def cossim_overlap(self, q, k, coord_q, coord_k):
        """ q, k: N * C * H * W
            coord_q, coord_k: N * 4 (x_upper_left, y_upper_left, x_lower_right, y_lower_right)
        """
        
        
        N, C, H, W = q.shape
        
        #print(q.shape)
        # [bs, feat_dim, 49]
        q = q.view(N, C, -1)
        k = k.view(N, C, -1)
        
        #print('loss->cossim_overlap, q, k: ',q.shape, k.shape)

        # generate center_coord, width, height
        # [1, 7, 7]
        x_array = torch.arange(0., float(W), dtype=coord_q.dtype, device=coord_q.device).view(1, 1, -1).repeat(1, H, 1)
        y_array = torch.arange(0., float(H), dtype=coord_q.dtype, device=coord_q.device).view(1, -1, 1).repeat(1, 1, W)
        # [bs, 1, 1]
        q_bin_width = ((coord_q[:, 2] - coord_q[:, 0]) / W).view(-1, 1, 1)
        q_bin_height = ((coord_q[:, 3] - coord_q[:, 1]) / H).view(-1, 1, 1)
        k_bin_width = ((coord_k[:, 2] - coord_k[:, 0]) / W).view(-1, 1, 1)
        k_bin_height = ((coord_k[:, 3] - coord_k[:, 1]) / H).view(-1, 1, 1)
        # [bs, 1, 1]
        q_start_x = coord_q[:, 0].view(-1, 1, 1)
        q_start_y = coord_q[:, 1].view(-1, 1, 1)
        k_start_x = coord_k[:, 0].view(-1, 1, 1)
        k_start_y = coord_k[:, 1].view(-1, 1, 1)

        # [bs, 1, 1]
        q_bin_diag = torch.sqrt(q_bin_width ** 2 + q_bin_height ** 2)
        k_bin_diag = torch.sqrt(k_bin_width ** 2 + k_bin_height ** 2)
        max_bin_diag = torch.max(q_bin_diag, k_bin_diag)

        # [bs, 7, 7]
        center_q_x = (x_array + 0.5) * q_bin_width + q_start_x
        center_q_y = (y_array + 0.5) * q_bin_height + q_start_y
        center_k_x = (x_array + 0.5) * k_bin_width + k_start_x
        center_k_y = (y_array + 0.5) * k_bin_height + k_start_y

        # [bs, 49, 49]
        dist_center = torch.sqrt((center_q_x.view(-1, H * W, 1) - center_k_x.view(-1, 1, H * W)) ** 2
                                 + (center_q_y.view(-1, H * W, 1) - center_k_y.view(-1, 1, H * W)) ** 2) / max_bin_diag
        pos_mask = (dist_center < self.pos_ratio*H).float().detach()

        # q: [bs, c, 14*14]
        # k: [bs, c, 14*14]
        logit = torch.bmm(q.transpose(1, 2), k)
        logit_norm = torch.bmm(q.transpose(1, 2).norm(dim=2, p=2, keepdim=True), \
                               k.norm(dim=1, p=2, keepdim=True))
        cosine_sim = logit/torch.clamp(logit_norm, min=1e-08)
        
        #print(q.shape, k.shape, coord_q.shape, coord_k.shape)
        #print(cosine_sim.shape, pos_mask.shape)
        #print()
        
        loss = (cosine_sim * pos_mask).sum(-1).sum(-1) / (pos_mask.sum(-1).sum(-1) + 1e-6)

        return loss
    
    def forward(self, p1, p2, z1, z2, coord_1, coord_2):
        #print('loss->forward, z1, z2: ',z1.shape, z2.shape)
        loss = -(self.cossim_overlap(p1, z2, coord_1, coord_2).mean() \
                 + self.cossim_overlap(p2, z1, coord_2, coord_1).mean()) * 0.5
        
        return loss
    

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)

        return k
