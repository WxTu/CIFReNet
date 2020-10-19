from torch import nn
from model.Backbone import mobilenet_v2
from model.Layers import *


class M2_semantic(nn.Module):
    def __init__(self, num_classes=19):
        super(M2_semantic, self).__init__()

        self.mod1 = mobilenet_v2[0]
        self.mod2 = mobilenet_v2[1]
        self.mod3 = nn.Sequential(mobilenet_v2[2], mobilenet_v2[3])
        self.mod4 = nn.Sequential(mobilenet_v2[4], mobilenet_v2[5], mobilenet_v2[6])
        self.mod5 = nn.Sequential(mobilenet_v2[7], mobilenet_v2[8], mobilenet_v2[9], mobilenet_v2[10])
        self.mod6 = nn.Sequential(mobilenet_v2[11], mobilenet_v2[12], mobilenet_v2[13])
        self.mod7 = nn.Sequential(mobilenet_v2[14], mobilenet_v2[15], mobilenet_v2[16])
        self.mod8 = mobilenet_v2[17]

        self.LRM = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            ABN(32),
            nn.Conv2d(32, 160, kernel_size=1),
            ABN(160)
        )

        self.multi_scale = MCIM(320, 320)

        self.attention = Attention_Block(160)

        self.final = nn.Sequential(
            nn.Conv2d(400, 256, kernel_size=1, stride=1, groups=16),
            ShuffleBlock(16),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        initialize_weights(self.LRM, self.attention, self.multi_scale, self.final)

    def forward(self, x):
        x_size = x.size()
        stg0 = self.mod1(x)

        stg1 = self.mod2(stg0)
        stg2 = self.mod3(stg1)
        stg3 = self.mod4(stg2)

        LRM = self.LRM(stg3)

        stg4 = self.mod5(stg3)
        stg5 = self.mod6(stg4)
        stg6 = self.mod7(stg5)

        attention = self.attention(stg6)

        stg6 = torch.mul(attention, LRM) + stg6

        stg7 = self.mod8(stg6)

        multi_scale = self.multi_scale(stg7)

        out = self.final(multi_scale)

        return F.upsample(out, x_size[2:], mode='bilinear')
