import torch.nn as nn


################### Region-Aware Conv ###################
class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        assert kH % 2 == 1
        self.mask.fill_(1)
        self.mask[:, :, kH//2, kH//2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)
    
class AnnularConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        assert kH % 2 == 1
        self.mask.fill_(0)
        self.mask[:, :, 0, :] = 1
        self.mask[:, :, -1, :] = 1
        self.mask[:, :, :, 0] = 1
        self.mask[:, :, :, -1] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class BifurcatedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        assert kH % 2 == 1
        self.mask.fill_(0)
        for i in range(kH):
            self.mask[:, :, i, i] = 1
            self.mask[:, :, i, kH-i-1] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class CrossedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        assert kH % 2 == 1
        self.mask.fill_(0)
        self.mask[:, :, kH//2, :] = 1
        self.mask[:, :, :, kH//2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class DilatedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        assert kH % 2 == 1
        self.mask.fill_(1)
        for i in range(1, kH, 2):
            self.mask[:, :, i, :] = 0
            self.mask[:, :, :, i] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)
