from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import ops
from torchvision.models import efficientnet

from network.PSAM import PSAModule
# from network.RepNeXt import ChunkConv


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


activation_layer = nn.ReLU(inplace=True)


# activation_layer = nn.LeakyReLU(0.1, inplace=True)


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__()
        middle_channels = in_channels // 2
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, middle_channels, 4, 2, 1, 0, bias=False),
            nn.BatchNorm2d(middle_channels),
            activation_layer,
            nn.Conv2d(middle_channels, num_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.conv(x)


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1,
                      padding=((kernel_size + 2 * (dilation - 1)) - 1) // 2, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            activation_layer
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            # nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            activation_layer
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, p):
        # print("in_channels:", in_channels, "out_channels:", out_channels)
        super(DecoderBlock, self).__init__()
        middle_channels = int(in_channels * 2)

        self.conv1 = Conv(in_channels, middle_channels, kernel_size=3, dilation=1)
        self.up = UpConv(middle_channels, middle_channels)
        self.conv2 = Conv(middle_channels, out_channels, kernel_size=3, dilation=1)

        # self.chunk_conv = ChunkConv(out_channels)

        self.drop = ops.DropBlock2d(p=p, block_size=3, inplace=True)

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.up(x)
        x = self.conv2(x)

        # y = self.chunk_conv(y)

        x = torch.cat([y, x], dim=1)
        x = self.drop(x)
        return x


class DecoderBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, p):
        super(DecoderBlock2, self).__init__()

        middle_channels = int(out_channels * 1)

        self.up = UpConv(in_channels, out_channels)

        self.conv1 = Conv(out_channels * 2, middle_channels, kernel_size=3, dilation=1)
        self.conv2 = Conv(middle_channels, out_channels * 2, kernel_size=3, dilation=1)

        self.drop = ops.DropBlock2d(p=p, block_size=3, inplace=False)

    def forward(self, x, y):
        x = self.up(x)
        x1 = torch.cat([y, x], dim=1)

        x1 = self.conv1(x1)
        x1 = self.conv2(x1)

        out = self.drop(x1)
        return out


class EfficientUNet(nn.Module):
    def __init__(self, num_classes, pretrain_backbone: bool = True, model_name: str = None, deep_supervision=False,
                 is_convert_onnx=False):
        super(EfficientUNet, self).__init__()
        if model_name == 'efficientnet_b0':
            backbone = efficientnet.efficientnet_b0(pretrained=pretrain_backbone)
            self.stage_out_channels = [16, 24, 40, 112, 320]
        elif model_name == 'efficientnet_b1':
            backbone = efficientnet.efficientnet_b1(pretrained=pretrain_backbone)
            self.stage_out_channels = [16, 24, 40, 112, 320]
        elif model_name == 'efficientnet_b2':
            backbone = efficientnet.efficientnet_b2(pretrained=pretrain_backbone)
            self.stage_out_channels = [16, 24, 48, 120, 352]
        elif model_name == 'efficientnet_b3':
            backbone = efficientnet.efficientnet_b3(pretrained=pretrain_backbone)
            self.stage_out_channels = [24, 32, 48, 136, 384]
        elif model_name == 'efficientnet_v2_s':
            backbone = efficientnet.efficientnet_v2_s(pretrained=pretrain_backbone)
            self.stage_out_channels = [24, 48, 64, 160, 1280]
        elif model_name == 'efficientnet_v2_m':
            backbone = efficientnet.efficientnet_v2_m(pretrained=pretrain_backbone)
            self.stage_out_channels = [24, 48, 80, 176, 512]
        else:
            exit(1)
        stage_indices = [1, 2, 3, 5, 7]
        return_layers = dict([(str(j), f"stage{i}") for i, j in enumerate(stage_indices)])
        self.backbone = IntermediateLayerGetter(backbone.features, return_layers=return_layers)
        drop = [0.2, 0.2, 0.2, 0.2]
        print(f"drop : {drop}, convert onnx : {is_convert_onnx}, deep_supervision : {deep_supervision}")
        self.up1 = DecoderBlock(self.stage_out_channels[4], self.stage_out_channels[3], drop[0])
        self.up2 = DecoderBlock(self.stage_out_channels[3] * 2, self.stage_out_channels[2], drop[1])
        self.up3 = DecoderBlock(self.stage_out_channels[2] * 2, self.stage_out_channels[1], drop[2])
        self.up4 = DecoderBlock(self.stage_out_channels[1] * 2, self.stage_out_channels[0], drop[3])
        self.outconv = OutConv(self.stage_out_channels[0] * 2, num_classes=num_classes)

        self.is_convert_onnx = is_convert_onnx
        self.deep_supervision = deep_supervision
        if self.deep_supervision:
            self.auxiliary0 = nn.Conv2d(self.stage_out_channels[0] * 2, num_classes, kernel_size=1)
            self.auxiliary1 = nn.Conv2d(self.stage_out_channels[1] * 2, num_classes, kernel_size=1)
            self.auxiliary2 = nn.Conv2d(self.stage_out_channels[2] * 2, num_classes, kernel_size=1)
            self.auxiliary3 = nn.Conv2d(self.stage_out_channels[3] * 2, num_classes, kernel_size=1)

        self.PSAM = PSAModule(self.stage_out_channels[4], self.stage_out_channels[4])
        # self.PPM = PPM(self.stage_out_channels[4], self.stage_out_channels[4] // 4, [2, 3, 5, 6])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.is_convert_onnx:
            x = x.permute(0, 3, 1, 2)  # rgb
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        backbone_out = self.backbone(x)
        # for i in range(5):
        #     print(i, backbone_out['stage{}'.format(str(i))].shape)
        # encoder
        e0 = backbone_out['stage0']
        e1 = backbone_out['stage1']
        e2 = backbone_out['stage2']
        e3 = backbone_out['stage3']
        e4 = backbone_out['stage4']

        # center
        e4 = self.PSAM(e4)
        # e4 = self.PPM(e4)

        # decoder
        d4 = self.up1(e4, e3)
        d3 = self.up2(d4, e2)
        d2 = self.up3(d3, e1)
        d1 = self.up4(d2, e0)
        out = self.outconv(d1)
        if self.is_convert_onnx:
            return out.permute(0, 2, 3, 1)
        if self.training and self.deep_supervision:
            # print(d1.shape, d2.shape, d3.shape, d4.shape)
            # print(self.auxiliary0.weight.shape, self.auxiliary1.weight.shape, self.auxiliary2.weight.shape, )
            aux_output0 = F.interpolate(self.auxiliary0(d1), scale_factor=2, mode='bilinear', align_corners=True)
            aux_output1 = F.interpolate(self.auxiliary1(d2), scale_factor=4, mode='bilinear', align_corners=True)
            aux_output2 = F.interpolate(self.auxiliary2(d3), scale_factor=8, mode='bilinear', align_corners=True)
            aux_output3 = F.interpolate(self.auxiliary3(d4), scale_factor=16, mode='bilinear', align_corners=True)
            # print(out.shape, aux_output0.shape, aux_output1.shape, aux_output2.shape, aux_output3.shape,
            #       aux_output3.shape)
            return {'out': out, 'aux_output0': aux_output0, 'aux_output1': aux_output1,
                    'aux_output2': aux_output2, 'aux_output3': aux_output3}

        return {'out': out}


if __name__ == '__main__':
    from torchsummary import summary

    model = EfficientUNet(num_classes=5, pretrain_backbone=True,
                          model_name='efficientnet_b2', deep_supervision=True).to("cuda")
    summary(model, (3, 224, 224))
