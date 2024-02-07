import torch
import torch.nn as nn
import torch.nn.functional as F

from . import regist_loss


eps = 1e-6

# ============================ #
#  Self-reconstruction loss    #
# ============================ #

@regist_loss
class self_L1():
    def __call__(self, input_data, model_output, data, module):
        output = model_output['recon_search']
        target_noisy = data['real_noisy_search']

        return F.l1_loss(output, target_noisy)  

@regist_loss
class self_L2():
    def __call__(self, input_data, model_output, data, module):
        output = model_output['recon_search']
        target_noisy = data['real_noisy_search']

        return F.mse_loss(output, target_noisy)
