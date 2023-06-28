import torch.nn as nn
import torch
from torch.nn import functional as F


class APL(nn.Module):
    def __init__(self, channel):
        super(APL, self).__init__()

        self.b0 = nn.Sequential(
            nn.AdaptiveMaxPool2d(13),
            nn.Conv2d(channel, channel, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True)
        )

        self.b1 = nn.Sequential(
            nn.AdaptiveMaxPool2d(9),
            nn.Conv2d(channel, channel, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True)
        )

        self.fus = nn.Sequential(
            nn.Conv2d(channel * 3, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True)
        )

    def forward(self, x):
        x_size = x.size()[2:]
        b0 = F.interpolate(self.b0(x), x_size, mode='bilinear', align_corners=True)
        b1 = F.interpolate(self.b1(x), x_size, mode='bilinear', align_corners=True)
        out = self.fus(torch.cat((b0, b1, x), 1))
        return out


class block1(nn.Module):
    def __init__(self, channels):
        super(block1, self).__init__()

        # self.cat = Channel_Att(channels)
        self.cat = CAT(channels)
        self.sat = SAT()

        self.d = CrossD(channels)

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.fusion_total = BasicConv2d(channels, channels, 3, 1, 1)

        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, infr, inft):
        infe = self.conv1(infr)
        inft = self.conv1(inft)
        inftotal = self.fusion_total(torch.mul(infe, inft) + torch.mul(inft, infe))
        infto = self.cat(inftotal) * inftotal
        inftod = self.d(inftotal)
        infin = self.conv(torch.cat((inftod, infto), 1))

        inres = self.sat(infin) * infin

        return inres


class block2(nn.Module):
    def __init__(self, channels):
        super(block2, self).__init__()

        self.mpltf = APL(channels)
        self.mpltrt = APL(channels)

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=1, padding=(0, 1),
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=1, padding=(1, 0),
                      groups=channels, bias=False)
        )

    def forward(self, fea, feart):
        infea = self.mpltf(fea)
        infeart = self.mpltrt(feart)

        sinrt = self.conv1(infeart - infea)

        sinrtgate = torch.sigmoid(sinrt)

        new_sinrt = (infeart - infea) * sinrtgate

        fin = self.conv1(infeart + new_sinrt)

        return fin, self.conv1(new_sinrt)


class block3(nn.Module):
    def __init__(self, channels):
        super(block3, self).__init__()

        self.naccr = REH(channels)

    def forward(self, rgb, t, nrgbt):
        nrgb, nt = self.naccr(rgb, t, nrgbt)

        return nrgb, nt


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class CrossD(nn.Module):
    def __init__(self, channel):
        super(CrossD, self).__init__()

        self.convk3d3 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=3, dilation=3,
                      groups=channel, bias=False)
        self.convk3d5 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=5, dilation=5,
                      groups=channel, bias=False)
        self.convk3d7 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=7, dilation=7,
                      groups=channel, bias=False)
        self.convk3d9 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=9, dilation=9,
                      groups=channel, bias=False)

        self.conv = nn.Sequential(
            nn.Conv2d(channel * 4, channel * 2, kernel_size=1),
            nn.BatchNorm2d(channel * 2),
            nn.ReLU(True),
            nn.Conv2d(channel * 2, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True)
        )

    def forward(self, x):
        c2 = self.convk3d3(x)
        c3 = self.convk3d5(x + c2)
        c4 = self.convk3d7(x + c3)
        c5 = self.convk3d9(x + c4)
        c2_2 = self.convk3d9(c2)
        c3_3 = self.convk3d7(c2_2 + c3)
        c4_4 = self.convk3d5(c3_3 + c4)
        c5_5 = self.convk3d3(c4_4 + c5)
        c2_22 = c2_2 + c3_3 + c4_4 + c5_5
        c3_33 = c3_3 + c4_4 + c5_5 + c2_2
        c4_44 = c4_4 + c5_5 + c3_3 + c2_2
        c5_55 = c5_5 + c4_4 + c2_2 + c3_3

        res = torch.cat((c2_22, c3_33, c4_44, c5_55), 1)

        return self.conv(res)


class CAT(nn.Module):
    def __init__(self, in_channels):
        super(CAT, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SAT(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAT, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class REH(nn.Module):
    def __init__(self, channel):
        super(REH, self).__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True),
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True),
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Conv2d(channel, 1, 3, padding=1),
        )
        self.channel = channel

    def forward(self, xr, xt, y):
        # print(y.shape)
        a = torch.sigmoid(-y)
        xr = a.expand(-1, self.channel, -1, -1).mul(xr)
        yr = self.conv2(xr)
        xt = a.expand(-1, self.channel, -1, -1).mul(xt)
        yt = self.conv2(xt)

        return yr, yt
