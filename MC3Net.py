import math
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from models.ModelTwo.One.MC3FDder import block1, block2, block3
from models.NET_Temp.convnext import convnext_small, LayerNorm, convnextt_small


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.vgg_r = convnext_small(True)
        self.vgg_t = convnext_small(True)
        self.vgg_rt = convnextt_small(False)

        self.layer0_r = nn.Sequential(self.vgg_r.downsample_layers[0], self.vgg_r.stages[0],
                                      LayerNorm(96, eps=1e-6, data_format="channels_first"))
        self.layer1_r = nn.Sequential(self.vgg_r.downsample_layers[1], self.vgg_r.stages[1],
                                      LayerNorm(192, eps=1e-6, data_format="channels_first"))
        self.layer2_r = nn.Sequential(self.vgg_r.downsample_layers[2], self.vgg_r.stages[2],
                                      LayerNorm(384, eps=1e-6, data_format="channels_first"))
        self.layer3_r = nn.Sequential(self.vgg_r.downsample_layers[3], self.vgg_r.stages[3],
                                      LayerNorm(768, eps=1e-6, data_format="channels_first"))

        self.layer0_t = nn.Sequential(self.vgg_t.downsample_layers[0], self.vgg_t.stages[0],
                                      LayerNorm(96, eps=1e-6, data_format="channels_first"))
        self.layer1_t = nn.Sequential(self.vgg_t.downsample_layers[1], self.vgg_t.stages[1],
                                      LayerNorm(192, eps=1e-6, data_format="channels_first"))
        self.layer2_t = nn.Sequential(self.vgg_t.downsample_layers[2], self.vgg_t.stages[2],
                                      LayerNorm(384, eps=1e-6, data_format="channels_first"))
        self.layer3_t = nn.Sequential(self.vgg_t.downsample_layers[3], self.vgg_t.stages[3],
                                      LayerNorm(768, eps=1e-6, data_format="channels_first"))

        self.layer0_rt = nn.Sequential(self.vgg_rt.downsample_layers[0], self.vgg_rt.stages[0],
                                       LayerNorm(96, eps=1e-6, data_format="channels_first"))
        self.layer1_rt = nn.Sequential(self.vgg_rt.downsample_layers[1], self.vgg_rt.stages[1],
                                       LayerNorm(192, eps=1e-6, data_format="channels_first"))
        self.layer2_rt = nn.Sequential(self.vgg_rt.downsample_layers[2], self.vgg_rt.stages[2],
                                       LayerNorm(384, eps=1e-6, data_format="channels_first"))
        self.layer3_rt = nn.Sequential(self.vgg_rt.downsample_layers[3], self.vgg_rt.stages[3],
                                       LayerNorm(768, eps=1e-6, data_format="channels_first"))

        self.fusion11 = block1(96)
        self.fusion12 = block1(192)
        self.fusion13 = block1(384)
        self.fusion14 = block1(768)

        self.fusion21 = block2(96)
        self.fusion22 = block2(192)
        self.fusion23 = block2(384)
        self.fusion24 = block2(768)

        self.fusion31 = block3(96)
        self.fusion32 = block3(192)
        self.fusion33 = block3(384)
        self.fusion34 = block3(768)

        self.upsam = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.reg_layer1 = nn.Sequential(
            nn.Conv2d(768, 384, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 192, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 96, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

    def forward(self, RGBT):
        image = RGBT[0]
        t = RGBT[0]
        rgbt = torch.cat((image, image), 1)

        conv1_vgg_r = self.layer0_r(image)
        conv1_vgg_t = self.layer0_t(t)
        conv1_vgg_rt = self.layer0_rt(rgbt)

        # print(conv1_vgg_r.shape,conv1_vgg_t.shape)
        f1 = self.fusion11(conv1_vgg_r, conv1_vgg_t)
        ff1, g1 = self.fusion21(f1, conv1_vgg_rt)
        rgb1, t1 = self.fusion31(conv1_vgg_r, conv1_vgg_t, g1)
        resr1 = rgb1 + conv1_vgg_r
        rest1 = t1 + conv1_vgg_t
        resrt1 = ff1 + conv1_vgg_rt

        conv2_vgg_r = self.layer1_r(resr1)
        conv2_vgg_t = self.layer1_t(rest1)
        conv2_vgg_rt = self.layer1_r(resrt1)

        f2 = self.fusion12(conv2_vgg_r, conv2_vgg_t)
        ff2, g2 = self.fusion22(f2, conv2_vgg_rt)
        rgb2, t2 = self.fusion32(conv2_vgg_r, conv2_vgg_t, g2)
        resr2 = rgb2 + conv2_vgg_r
        rest2 = t2 + conv2_vgg_t
        resrt2 = ff2 + conv2_vgg_rt

        conv3_vgg_r = self.layer2_r(resr2)
        conv3_vgg_t = self.layer2_t(rest2)
        conv3_vgg_rt = self.layer2_r(resrt2)

        f3 = self.fusion13(conv3_vgg_r, conv3_vgg_t)
        ff3, g3 = self.fusion23(f3, conv3_vgg_rt)
        rgb3, t3 = self.fusion33(conv3_vgg_r, conv3_vgg_t, g3)
        resr3 = rgb3 + conv3_vgg_r
        rest3 = t3 + conv3_vgg_t
        resrt3 = ff3 + conv3_vgg_rt

        conv4_vgg_r = self.layer3_r(resr3)
        conv4_vgg_t = self.layer3_t(rest3)
        conv4_vgg_rt = self.layer3_r(resrt3)

        f4 = self.fusion14(conv4_vgg_r, conv4_vgg_t)
        ff4, g4 = self.fusion24(f4, conv4_vgg_rt)
        rgb4, t4 = self.fusion34(conv4_vgg_r, conv4_vgg_t, g4)
        resr4 = rgb4 + conv4_vgg_r
        rest4 = t4 + conv4_vgg_t
        resrt4 = ff4 + conv4_vgg_rt
        fin = resr4 + rest4 + resrt4

        fino = self.upsam(fin)
        fin = self.reg_layer1(fino)
        # print(fin)
        # rgb4 = F.interpolate(rgb4, (fin.size()[2], fin.size()[3]))
        return fin


if __name__ == '__main__':
    rgb = torch.randn(1, 3, 480, 640)
    depth = torch.randn(1, 3, 480, 640)
    # rgb = torch.randn(1, 3, 256, 256)
    # depth = torch.randn(1, 3, 256, 256)
    # a = torch.randn(1, 3, 32, 32)
    # b = F.interpolate(rgb, (a.size()[2], a.size()[3]))
    # print(b.shape)
    model = Net()
    res = model([rgb, depth])
    print(res.shape)
    # from models.ModelTwo.One.canshu.utils import compute_speed
    # from ptflops import get_model_complexity_info
    # with torch.cuda.device(0):
    #     net = Net()
    #     flops, params = get_model_complexity_info(net, (3, 480, 640), as_strings=True, print_per_layer_stat=False)
    #     print('Flops:' + flops)
    #     print('Params:' + params)
    #
    # compute_speed(net, input_size=(1, 3, 480, 640), iteration = 500)
