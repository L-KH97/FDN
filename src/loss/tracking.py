import torch
import torch.nn as nn
import torch.nn.functional as F

from . import regist_loss

from ..pysot.models.backbone import get_backbone
from ..pysot.models.head import get_rpn_head
from ..pysot.utils.model_load import load_pretrain

eps = 1e-6

# ====================== #
#      tracking loss     #
# ====================== #

@regist_loss
class tracking_loss():
    def __init__(self):
        super(tracking_loss, self).__init__()
        self.model = Tracking_Model().cuda()
        self.model = load_pretrain(self.model, './experiments/SiamRPN_alex/model.pth').cuda().eval()

        for _, params in self.model.named_parameters():
            params.requires_grad = False
    
    def __call__(self, input_data, model_output, data, module):
        cls, loc = self.model(model_output)
        cls = self.log_softmax(cls)

        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()

        return 1.0 * self.cls_loss(cls, label_cls) + \
               1.2 * self.loc_loss(loc, label_loc, label_loc_weight)
    
    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def get_cls_loss(self, pred, label, select):
        if len(select.size()) == 0 or \
                select.size() == torch.Size([0]):
            return 0
        pred = torch.index_select(pred, 0, select)
        label = torch.index_select(label, 0, select)
        return F.nll_loss(pred, label)

    def cls_loss(self, pred, label):
        pred = pred.view(-1, 2)
        label = label.view(-1)
        pos = label.data.eq(1).nonzero().squeeze().cuda()
        neg = label.data.eq(0).nonzero().squeeze().cuda()
        loss_pos = self.get_cls_loss(pred, label, pos)
        loss_neg = self.get_cls_loss(pred, label, neg)
        return loss_pos * 0.5 + loss_neg * 0.5
    
    def loc_loss(self, pred_loc, label_loc, loss_weight):
        b, _, sh, sw = pred_loc.size()
        pred_loc = pred_loc.view(b, 4, -1, sh, sw)
        diff = (pred_loc - label_loc).abs()
        diff = diff.sum(dim=1).view(b, -1, sh, sw)
        loss = diff * loss_weight
        return loss.sum().div(b)

class Tracking_Model(nn.Module):
    def __init__(self):
        super(Tracking_Model, self).__init__()
        self.backbone = get_backbone('alexnetlegacy',
                                     width_mult=1.0)

        # build rpn head
        self.rpn_head = get_rpn_head('DepthwiseRPN',
                                     anchor_num=5,
                                     in_channels=256,
                                     out_channels=256)

    
    def forward(self, data):
        """ only used in training
        """
        template = data['recon_template']
        search = data['recon_search']

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
  
        cls, loc = self.rpn_head(zf, xf)

        return cls, loc
