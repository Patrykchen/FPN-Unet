import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import torch

input_size = (448, 448)


class Bottleneck(nn.Module):
    expansion = 4   # 通道倍增数

    def __init__(self, in_planes, planes, stride=1, downsample = None): # 通道数 --> 输出通道数 * 4
        super(Bottleneck, self).__init__()

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_planes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, stride, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, self.expansion * planes, 1, bias=False),
            nn.BatchNorm2d(self.expansion * planes),
        )

        self.downsample = downsample    # 下采样函数，默认为None

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, layers):
        super(FPN, self).__init__()
        self.inplanes = 64
        # 第一个卷积层，通道扩充 + 尺寸重构
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)  # 3个通道到64个通道，尺寸下降一半
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)    # 3*3，步长为2，padding为1

        # 每一个都是resnet的bottleneck，layers是每个小模块的层数，第一个参数 * 4为最后的输出通道数，层间的输入通道数用变量self.inplanes控制
        self.layer1 = self._make_layer(64, layers[0])   # C2
        self.layer2 = self._make_layer(128, layers[1], 2)   # C3
        self.layer3 = self._make_layer(256, layers[2], 2)   # C4
        self.layer4 = self._make_layer(512, layers[3], 2)   # C5

        # 对C2到C5层进行横向链接（Lateral Connection），通道数均变为256
        self.latlayer1 = nn.Conv2d(2048, 256, 1, 1, 0)  # C5，即为直接需要输出的P5
        self.latlayer2 = nn.Conv2d(1024, 256, 1, 1, 0)  # C4
        self.latlayer3 = nn.Conv2d(512, 256, 1, 1, 0)   # C3
        self.latlayer4 = nn.Conv2d(256, 256, 1, 1, 0)   # C2

        # 3*3的积融合，目的是消除上采样过程带来的重叠效应
        self.smooth = nn.Conv2d(256, 1, 3, 1, 1)    # 256



    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != Bottleneck.expansion * planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, Bottleneck.expansion * planes, 1, stride, bias=False),
                nn.BatchNorm2d(Bottleneck.expansion * planes)
            )
        ###初始化需要一个list，代表左侧网络ResNet每一个阶段的Bottleneck的数量
        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)
        ###自上而下的上采样模块


    def _upsample_add(self, x, y):
        _, _, H, W = y.shape
        return F.upsample(x, size=(H, W), mode='nearest') + y


    def forward(self, x):
        # 自下而上
        c1 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # 自上而下
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4)) # 同时完成上采样以及横向链接
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p2 = self._upsample_add(p3, self.latlayer4(c2)) # 现在是1/4

        p2 = F.upsample(p2, size=input_size, mode='bilinear')

        # 卷积融合，平滑处理
        p4 = self.smooth(p4)
        p3 = self.smooth(p3)
        p2 = self.smooth(p2)

        # p2 = F.interpolate(p2, size=input_size, mode='nearest')
        # p3 = F.interpolate(p3, size=input_size, mode='nearest')
        # p4 = F.interpolate(p4, size=input_size, mode='nearest')
        # p5 = F.interpolate(p5, size=input_size, mode='nearest')
        #
        # p2 = p2 + p3 + p4 + p5

        return p2, p3, p4, p5