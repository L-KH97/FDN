import torch
import torch.nn as nn

from . import regist_model
from .RAConv import AnnularConv2d, BifurcatedConv2d, CrossedConv2d, DilatedConv2d, CentralMaskedConv2d
from .Trans import LayerNorm, SelfAttention, CrossAttention, FeedForward


@regist_model
class RTN(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_dim=128, num_heads=8, num_module=3, ffn_factor=2):
        '''
        Args:
            in_ch      : number of input channel
            out_ch     : number of output channel
            base_dim   : number of base channel
        '''
        super().__init__()

        assert base_dim%2 == 0, "base channel should be divided with 2"

        ly = []
        ly += [ nn.Conv2d(in_ch, base_dim//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ Downsample(int(base_dim//2)) ]
        self.head = nn.Sequential(*ly)

        self.body = RAFT(3, base_dim, num_heads, ffn_factor, num_module)

        ly = []
        ly += [ Upsample(int(base_dim)) ]
        ly += [ nn.Conv2d(base_dim//2, base_dim//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_dim//2, out_ch,     kernel_size=1) ]
        self.tail = nn.Sequential(*ly)

    def forward(self, x):
        return self.tail(self.body(self.head(x)))

    def _initialize_weights(self):
        # Liyong version
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)

class RAFT(nn.Module):
    def __init__(self, stride, base_dim=128, num_heads=8, num_module=3, ffn_factor=2):
        '''
        Args:
            base_dim   : number of base channel
            att_dim    : number of attention
            ffn_factor : expansion factor of feed-forward network
        '''
        super().__init__()

        self.branchA = AC_branch(stride, base_dim, num_module)
        self.branchB = BC_branch(stride, base_dim, num_module)
        self.branchC = CC_branch(stride, base_dim, num_module)
        self.branchD = DC_branch(stride, base_dim, num_module)

        ly = []
        ly += [ nn.Conv2d(int(base_dim*4), int(base_dim*2), kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(int(base_dim*2), base_dim, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        self.bridgeRG = nn.Sequential(*ly)

        self.branchM1 = CMC_branch(int(stride-1), base_dim, num_module)
        self.branchM2 = CMC_branch(stride, base_dim, num_module)

        ly = []
        ly += [ nn.Conv2d(int(base_dim*2), base_dim, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        self.bridgeCM = nn.Sequential(*ly)

        self.norm1 = LayerNorm(base_dim)
        self.satt = SelfAttention(base_dim, num_heads)
        self.norm2 = LayerNorm(base_dim)
        self.ffn1 = FeedForward(base_dim, ffn_factor)

        self.norm3 = LayerNorm(base_dim)
        self.catt = CrossAttention(base_dim, num_heads)
        self.norm4 = LayerNorm(base_dim)
        self.ffn2 = FeedForward(base_dim, ffn_factor)

    def forward(self, x):
        xRG = self.bridgeRG(torch.cat([self.branchA(x), self.branchB(x), self.branchC(x), self.branchD(x)], dim=1))

        xRG = xRG + self.satt(self.norm1(xRG))
        xRG = xRG + self.ffn1(self.norm2(xRG))

        xCM = self.bridgeCM(torch.cat([self.branchM1(x), self.branchM2(x)], dim=1))

        xCM = xCM + self.catt(self.norm3(xCM), xRG)
        xCM = xCM + self.ffn2(self.norm4(xCM))

        return xCM

class DCl(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)

class AC_branch(nn.Module):
    def __init__(self, stride, in_ch, num_module=3):
        super().__init__()

        ly = []
        ly += [ AnnularConv2d(in_ch, in_ch, kernel_size=2*stride-1, stride=1, padding=stride-1) ]
        ly += [ nn.ReLU(inplace=True) ]

        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]

        ly += [ DCl(stride, in_ch) for _ in range(num_module) ]

        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        
        self.branch = nn.Sequential(*ly)

    def forward(self, x):
        return self.branch(x)

class BC_branch(nn.Module):
    def __init__(self, stride, in_ch, num_module=3):
        super().__init__()

        ly = []
        ly += [ BifurcatedConv2d(in_ch, in_ch, kernel_size=2*stride-1, stride=1, padding=stride-1) ]
        ly += [ nn.ReLU(inplace=True) ]

        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]

        ly += [ DCl(stride, in_ch) for _ in range(num_module) ]

        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        
        self.branch = nn.Sequential(*ly)

    def forward(self, x):
        return self.branch(x)

class CC_branch(nn.Module):
    def __init__(self, stride, in_ch, num_module=3):
        super().__init__()

        ly = []
        ly += [ CrossedConv2d(in_ch, in_ch, kernel_size=2*stride-1, stride=1, padding=stride-1) ]
        ly += [ nn.ReLU(inplace=True) ]

        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]

        ly += [ DCl(stride, in_ch) for _ in range(num_module) ]

        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        
        self.branch = nn.Sequential(*ly)

    def forward(self, x):
        return self.branch(x)

class DC_branch(nn.Module):
    def __init__(self, stride, in_ch, num_module=3):
        super().__init__()

        ly = []
        ly += [ DilatedConv2d(in_ch, in_ch, kernel_size=2*stride-1, stride=1, padding=stride-1) ]
        ly += [ nn.ReLU(inplace=True) ]

        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]

        ly += [ DCl(stride, in_ch) for _ in range(num_module) ]

        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        
        self.branch = nn.Sequential(*ly)

    def forward(self, x):
        return self.branch(x)

class CMC_branch(nn.Module):
    def __init__(self, stride, in_ch, num_module=3):
        super().__init__()

        ly = []
        ly += [ CentralMaskedConv2d(in_ch, in_ch, kernel_size=2*stride-1, stride=1, padding=stride-1) ]
        ly += [ nn.ReLU(inplace=True) ]

        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]

        ly += [ DCl(stride, in_ch) for _ in range(num_module) ]

        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        
        self.branch = nn.Sequential(*ly)

    def forward(self, x):
        return self.branch(x)

@regist_model
class RFN(nn.Module):
    def __init__(self, in_ch=6, out_ch=3, refine_dim=32, num_heads=8, ffn_factor=2):
        '''
        Args:
            in_ch      : number of input channel
            out_ch     : number of output channel
            base_dim   : number of base channel
        '''
        super().__init__()

        assert refine_dim%2 == 0, "base channel should be divided with 2"

        ly = []
        ly += [ nn.Conv2d(in_ch, refine_dim//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(refine_dim//2, refine_dim, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        self.head = nn.Sequential(*ly)

        self.body = DAFT(refine_dim, num_heads, ffn_factor)

        ly = []
        ly += [ nn.Conv2d(refine_dim, refine_dim, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(refine_dim, refine_dim//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(refine_dim//2, out_ch, kernel_size=1) ]
        self.tail = nn.Sequential(*ly)

    def forward(self, x, y):
        return self.tail(self.body(self.head(torch.cat((x, y), dim=1))))

class DAFT(nn.Module):
    def __init__(self, refine_dim=32, num_heads=8, ffn_factor=2):
        '''
        Args:
            base_dim   : number of base channel
            att_dim    : number of attention
            ffn_factor : expansion factor of feed-forward network
        '''
        super().__init__()

        self.norm1 = LayerNorm(refine_dim)
        self.satt = SelfAttention(refine_dim, num_heads)
        self.norm2 = LayerNorm(refine_dim)
        self.ffn1 = FeedForward(refine_dim, ffn_factor)


    def forward(self, x):

        x = x + self.satt(self.norm1(x))
        x = x + self.ffn1(self.norm2(x))

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)