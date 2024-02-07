
import random
import torch.nn as nn
import torch.nn.functional as F

from ..util.util import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling
from . import regist_model
from .TN import RTN, RFN

@regist_model
class FDN(nn.Module):
    '''
    Focal Denoising Network (FDN)
    '''
    def __init__(self, pd_train=[2, 3, 4, 5], pd_test=2, pd_pad=2, in_ch=3, 
                 base_dim=128, refine_dim=32, num_heads=8, num_module=3, ffn_factor=2):
        '''
        Args:
            pd             : PD stride factor
            pd_pad         : pad size between sub-images by PD process
            in_ch          : number of input image channel
            base_dim       : number of base channel
        '''
        super().__init__()

        # network hyper-parameters
        self.pd_train = pd_train
        self.pd_test = pd_test
        self.pd_pad = pd_pad

        # define network
        self.rtn = RTN(in_ch, in_ch, base_dim, num_heads, num_module, ffn_factor)
        self.rfn = RFN(int(in_ch*2), in_ch, refine_dim, num_heads, ffn_factor)


    def forward(self, img_z, img_x):
        img_denoised_z, _ = self.denoise(img_z)
        img_denoised_x, img_tmp_x = self.denoise(img_x)

        return {'img_template': img_denoised_z, 'img_search': img_denoised_x, 'img_tmp': img_tmp_x}
    
    def denoise(self, img):
        if self.training:
            pd = random.choice(self.pd_train)
        else:
            pd = self.pd_test

        img, h1, h2, w1, w2 = self.srd_pad(img, pd)
        
        pd_img = pixel_shuffle_down_sampling(img, f=pd, pad=self.pd_pad)

        pd_img_denoised = self.rtn(pd_img)

        pu_img = pixel_shuffle_up_sampling(pd_img_denoised, f=pd, pad=self.pd_pad)

        img_denoised = self.rfn(pu_img, img)

        return img_denoised[:, :, h1:-h2, w1:-w2], pu_img[:, :, h1:-h2, w1:-w2]

    def srd_pad(self, img, pd):
        b,c,h,w = img.shape

        h1 = (pd-h%pd)//2
        h2 = h1 + (pd-h%pd)%2
        w1 = (pd-w%pd)//2
        w2 = w1 + (pd-w%pd)%2

        if h2 != 0:
            img = F.pad(img, (0, 0, h1, h2), mode='constant', value=0)
        if w2 != 0:
            img = F.pad(img, (w1, w2, 0, 0), mode='constant', value=0)
        
        return img, h1, h2, w1, w2
