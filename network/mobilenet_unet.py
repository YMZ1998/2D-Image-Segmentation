from collections import OrderedDict
from functools import partial
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import models, ops
from torchvision.models import mobilenet_v3_large, mobilenet_v2, mobilenet_v3_small
from network.ACBlock import ACBlock
from network.FRN import FRN

nonlinearity = partial(F.relu, inplace=True)


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes, r=2):
        super(OutConv, self).__init__(
            nn.ConvTranspose2d(in_channels, in_channels // r, 3, 2, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Conv2d(in_channels // r, in_channels // r, 3, padding=1),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels // r, num_classes, 3, 1, 1)
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, r=1):
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
        # self.norm3 = FRN(n_filters)
        self.relu3 = nn.LeakyReLU(0.1, inplace=True)
        p = 0.2
        self.drop = ops.DropBlock2d(p=p, block_size=3)
        # self.drop2 = ops.DropBlock2d(p=p, block_size=3)

        # self.norm4 = FRN(n_filters * 2)

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        # x = self.drop(x)
        # y = self.drop2(y)
        x = torch.cat([y, x], dim=1)
        # x = self.norm4(x)
        x = self.drop(x)

        return x


class IntermediateLayerGetter(nn.ModuleDict):
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class MobileV3UNet(nn.Module):
    def __init__(self, num_classes, pretrain_backbone: bool = False):
        super(MobileV3UNet, self).__init__()
        backbone = mobilenet_v3_large(pretrained=pretrain_backbone)
        backbone = backbone.features

        stage_indices = [1, 3, 6, 12, 15]
        self.stage_out_channels = [backbone[i].out_channels for i in stage_indices]
        return_layers = dict([(str(j), f"stage{i}") for i, j in enumerate(stage_indices)])
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        self.up1 = DecoderBlock(self.stage_out_channels[4], self.stage_out_channels[3])
        self.up2 = DecoderBlock(self.stage_out_channels[3] * 2, self.stage_out_channels[2])
        self.up3 = DecoderBlock(self.stage_out_channels[2] * 2, self.stage_out_channels[1])
        self.up4 = DecoderBlock(self.stage_out_channels[1] * 2, self.stage_out_channels[0])
        self.outconv = OutConv(self.stage_out_channels[0] * 2, num_classes=num_classes)

        # from network.PSAM import PPM
        # self.PPM = PPM(self.stage_out_channels[4], self.stage_out_channels[4] // 8, [2, 3, 5, 6])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # input_shape = x.shape[-2:]
        backbone_out = self.backbone(x)
        # encoder
        e0 = backbone_out['stage0']
        e1 = backbone_out['stage1']
        e2 = backbone_out['stage2']
        e3 = backbone_out['stage3']
        e4 = backbone_out['stage4']

        # center
        # e4 = self.PPM(e4)

        # decoder
        d4 = self.up1(e4, e3)
        d3 = self.up2(d4, e2)
        d2 = self.up3(d3, e1)
        d1 = self.up4(d2, e0)
        out = self.outconv(d1)

        return {"out": out}


class MobileV2UNet(nn.Module):
    def __init__(self, num_classes, pretrain_backbone: bool = False):
        super(MobileV2UNet, self).__init__()
        backbone = mobilenet_v2(pretrained=pretrain_backbone)

        # if pretrain_backbone:
        #     #     # 载入mobilenetv3 large backbone预训练权重
        #     #     # https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth
        #     backbone.load_state_dict(torch.load("./network/mobilenet_v3_large.pth", map_location='cpu'))

        backbone = backbone.features

        stage_indices = [1, 3, 6, 13, 16]
        self.stage_out_channels = [backbone[i].out_channels for i in stage_indices]
        return_layers = dict([(str(j), f"stage{i}") for i, j in enumerate(stage_indices)])
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        self.up1 = DecoderBlock(self.stage_out_channels[4], self.stage_out_channels[3])
        self.up2 = DecoderBlock(self.stage_out_channels[3] * 2, self.stage_out_channels[2])
        self.up3 = DecoderBlock(self.stage_out_channels[2] * 2, self.stage_out_channels[1])
        self.up4 = DecoderBlock(self.stage_out_channels[1] * 2, self.stage_out_channels[0])
        self.outconv = OutConv(self.stage_out_channels[0] * 2, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_shape = x.shape[-2:]
        backbone_out = self.backbone(x)
        # encoder
        e0 = backbone_out['stage0']
        e1 = backbone_out['stage1']
        e2 = backbone_out['stage2']
        e3 = backbone_out['stage3']
        e4 = backbone_out['stage4']

        # decoder
        d4 = self.up1(e4, e3)
        d3 = self.up2(d4, e2)
        d2 = self.up3(d3, e1)
        d1 = self.up4(d2, e0)
        out = self.outconv(d1)
        return {"out": out}
