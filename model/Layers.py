from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % groups == 0)
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)


class ABN(nn.Sequential):
    def __init__(self, num_features):
        super(ABN, self).__init__(OrderedDict([
            ("bn", nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True)),
            ("act", nn.PReLU(num_features))
        ]))


class DSP(nn.Module):
    def __init__(self, inplanes, outplanes, c_tag=0.2, groups=4, dilation=(1, 2, 3, 4)):
        super(DSP, self).__init__()

        self.out_c = round(c_tag * outplanes)

        self.down = nn.Sequential(
            nn.Conv2d(inplanes, self.out_c, 1, stride=1, groups=groups, bias=False),
            ABN(self.out_c)
        )

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.out_c, self.out_c, 1, stride=1),
            nn.BatchNorm2d(self.out_c, eps=1e-05, momentum=0.1, affine=True)
        )

        self.branch_1 = nn.Sequential(
            nn.Conv2d(self.out_c, self.out_c, kernel_size=3, padding=dilation[0], dilation=dilation[0],
                      groups=self.out_c, bias=False),
            ABN(self.out_c),
            nn.Conv2d(self.out_c, self.out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_c, eps=1e-05, momentum=0.1, affine=True)
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(self.out_c, self.out_c, kernel_size=3, padding=dilation[1], dilation=dilation[1],
                      groups=self.out_c, bias=False),
            ABN(self.out_c),
            nn.Conv2d(self.out_c, self.out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_c, eps=1e-05, momentum=0.1, affine=True)
        )
        self.branch_3 = nn.Sequential(
            nn.Conv2d(self.out_c, self.out_c, kernel_size=3, padding=dilation[2], dilation=dilation[2],
                      groups=self.out_c, bias=False),
            ABN(self.out_c),
            nn.Conv2d(self.out_c, self.out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_c, eps=1e-05, momentum=0.1, affine=True)
        )

        self.branch_4 = nn.Sequential(
            nn.Conv2d(self.out_c, self.out_c, kernel_size=3, padding=dilation[3], dilation=dilation[3],
                      groups=self.out_c, bias=False),
            ABN(self.out_c),
            nn.Conv2d(self.out_c, self.out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_c, eps=1e-05, momentum=0.1, affine=True)
        )

        self.groups = groups

        self.module_act = nn.PReLU(outplanes)

    def forward(self, x):
        input_x = x
        x_size = x.size()
        x = self.down(x)
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)
        branch_4 = self.branch_4(x)
        pool = F.upsample(self.pool(x), x_size[2:], mode="bilinear")
        out = channel_shuffle(torch.cat((branch_1, branch_2, branch_3, branch_4, pool), 1), self.groups)

        if out.size() == input_x.size():
            out = out + input_x
        return self.module_act(out)


class MCIM(nn.Module):
    def __init__(self, in_chs=320, out_chs=320):
        super(MCIM, self).__init__()

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_chs, 80, 1, stride=1),
            nn.BatchNorm2d(80),
            nn.LeakyReLU(0.1)
        )

        self.conv_small = DSP(in_chs, out_chs, dilation=(1, 2, 3, 5))

        self.conv_middle = DSP(out_chs, out_chs, dilation=(7, 9, 11, 13))

        self.conv_larger = DSP(out_chs, out_chs, dilation=(17, 19, 21, 23))

    def forward(self, x):
        x_size = x.size()
        small = self.conv_small(x)
        middle = self.conv_middle(small)
        larger = self.conv_larger(middle)
        pool = F.upsample(self.pool(x), x_size[2:], mode="bilinear")
        output = small + middle + larger
        output = torch.cat([output, pool], 1)
        return output


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class Attention_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Attention_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel // reduction), channel),
                                                nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()
        chn_se = self.avg_pool(x).view(bahs, chs)
        return self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
