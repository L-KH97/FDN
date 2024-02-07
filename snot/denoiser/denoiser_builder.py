import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image

import os
from src.util.config_parse import ConfigParser
from src.trainer import get_trainer_class as FDN


class Denoiser():
    def __init__(self, model, args):
        super(Denoiser, self).__init__()
        
        self.model = model
        self.thr = 192

    def denoise(self, img):

        input_ = img

        h,w = input_.shape[2], input_.shape[3]
        if h<self.thr:
            H = W = self.thr
            padh = H-h
            padw = W-w
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        restored = self.model.test_track(input_)

        return restored[:,:,:h,:w]
    
    def single_denoise(self, img):
        name = img.split('/')[-1].split('.')[0]
        img = Image.open(img)
        img = (np.asarray(img)/255.0)
        img = torch.from_numpy(img).float()
        img = img.permute(2,0,1)
        input_ = img.cuda().unsqueeze(0)

        restored = self.model.test_track(input_)

        torchvision.utils.save_image(restored, os.path.join(name+'_DN.jpg'))



def build_denoiser(args):

    assert args.config is not None, 'config file path is needed'
    if args.session_name is None:
        args.session_name = args.config # set session name to config file name

    cfg = ConfigParser(args)

    # device setting
    if cfg['gpu'] is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu']

    # intialize trainer
    model = FDN(cfg['trainer'])(cfg)
    model._before_test(dataset_load=False)

    return Denoiser(model, args)

