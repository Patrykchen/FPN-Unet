import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url


class VGG(nn.Module):
    cfg = [64, 64, 'MaxPooling', 128, 128, 'MaxPooling', 256, 256, 256, 'MaxPooling', 512, 512, 512, 'MaxPooling', 512,
           512, 512, 'MaxPooling']

    def __init__(self, num_classes=1000):
        super(VGG, self).__init__()
        self.features = self._make_layer(self.cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def _make_layer(self, cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == 'MaxPooling':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def VGG16(pretrained=True):
    model = VGG()
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth",
                                              model_dir="./model_data")
        model.load_state_dict(state_dict)

    # 删除VGG网络中不需要的部分
    del model.avgpool
    del model.classifier

    return model


class Upadd(nn.Module):
    def __init__(self, in_size, out_size):
        super(Upadd, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self, num_classes=2):
        super(Unet, self).__init__()
        self.vgg = VGG16()
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]
        # upsampling
        self.up_concat4 = Upadd(in_filters[3], out_filters[3])
        self.up_concat3 = Upadd(in_filters[2], out_filters[2])
        self.up_concat2 = Upadd(in_filters[1], out_filters[1])
        self.up_concat1 = Upadd(in_filters[0], out_filters[0])

        # final conv (without any concat)
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        feat1 = self.vgg.features[:4](inputs)
        feat2 = self.vgg.features[4:9](feat1)
        feat3 = self.vgg.features[9:16](feat2)
        feat4 = self.vgg.features[16:23](feat3)
        feat5 = self.vgg.features[23:-1](feat4)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        result = self.final(up1)

        return result
