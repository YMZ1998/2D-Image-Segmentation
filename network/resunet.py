from functools import partial
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import models, ops

nonlinearity = partial(F.relu, inplace=True)


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes, r=2):
        super(OutConv, self).__init__(
            nn.ConvTranspose2d(in_channels, in_channels // r, 3, 2, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels // r, num_classes, 3, 1, 1)
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, r=2):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // r, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // r)
        # self.norm1 = FRN(in_channels // r)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels // r, in_channels // r, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // r)
        self.relu2 = nn.LeakyReLU(0.1, inplace=True)

        self.conv3 = nn.Conv2d(in_channels // r, n_filters, 3, 1, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.LeakyReLU(0.1, inplace=True)
        p = 0.2
        self.drop = ops.DropBlock2d(p=p, block_size=3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        # x = torch.cat([y, x], dim=1)
        # x = self.drop(x)

        return x


class ResUNet(nn.Module):
    def __init__(self, num_classes, pretrain_backbone: bool = False):
        super(ResUNet, self).__init__()
        self.stage_out_channels = [64, 64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.up1 = DecoderBlock(self.stage_out_channels[4], self.stage_out_channels[3])
        self.up2 = DecoderBlock(self.stage_out_channels[3], self.stage_out_channels[2])
        self.up3 = DecoderBlock(self.stage_out_channels[2], self.stage_out_channels[1])
        self.up4 = DecoderBlock(self.stage_out_channels[1], self.stage_out_channels[0])
        self.outconv = OutConv(self.stage_out_channels[0], num_classes=num_classes)

        from network.PSAM import PPM
        self.PPM = PPM(self.stage_out_channels[4], self.stage_out_channels[4] // 8, [2, 3, 5, 6])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e0 = self.firstmaxpool(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # center
        # e4 = self.PPM(e4)

        # decoder
        d4 = self.up1(e4) + e3
        d3 = self.up2(d4) + e2
        d2 = self.up3(d3) + e1
        d1 = self.up4(d2)
        out = self.outconv(d1)

        return {"out": out}


if __name__ == '__main__':
    model = ResUNet(num_classes=3, pretrain_backbone=True).to("cuda")
    summary(model, (3, 192, 192))
