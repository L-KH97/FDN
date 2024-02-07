import torch

from . import regist_trainer
from .base import BaseTrainer
from ..model import get_model_class


@regist_trainer
class Trainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    @torch.no_grad()
    def validation(self):
        # set denoiser
        self._set_denoiser()

        # make directories for image saving
        img_save_path = 'img/val_%03d' % self.epoch
        self.file_manager.make_dir(img_save_path)

        # validation
        psnr, ssim = self.test_dataloader_process(  dataloader    = self.val_dataloader['dataset'],
                                                    add_con       = 0.  if not 'add_con' in self.val_cfg else self.val_cfg['add_con'],
                                                    floor         = False if not 'floor' in self.val_cfg else self.val_cfg['floor'],                                                    
                                                    img_save_path = img_save_path,
                                                    img_save      = self.val_cfg['save_image'])

    def _set_module(self):
        module = {}
        if self.cfg['model']['kwargs'] is None:
            module['denoiser'] = get_model_class(self.cfg['model']['type'])()
        else:   
            module['denoiser'] = get_model_class(self.cfg['model']['type'])(**self.cfg['model']['kwargs'])
        return module
        
    def _set_optimizer(self):
        optimizer = {}
        for key in self.module:
            optimizer[key] = self._set_one_optimizer(opt        = self.train_cfg['optimizer'], 
                                                     parameters = self.module[key].parameters(), 
                                                     lr         = float(self.train_cfg['init_lr']))
        return optimizer

    def _forward_fn(self, module, loss, data):
        # forward
        input_data = [data['dataset'][arg] for arg in self.cfg['model_input']]
        denoised_img = module['denoiser'](*input_data)
        model_output = {'recon_template': denoised_img['img_template'],
                        'recon_search': denoised_img['img_search'],
                        'recon_tmp': denoised_img['img_tmp']}

        # get losses
        losses, tmp_info = loss(input_data, model_output, data['dataset'], module, \
                                    ratio=(self.epoch-1 + (self.iter-1)/self.max_iter)/self.max_epoch)

        return losses, tmp_info
        