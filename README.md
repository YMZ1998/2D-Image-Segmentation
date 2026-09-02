# [2D-Image-Segmentation](https://github.com/li-jin-1998/2D-Image-Segmentation)

Some implementation of segmentation for rgb and rgb-d images.

## Usage

export environment

```
conda activate AI
conda env export -n AI > myenv.yml
```

create environment

```
conda env create -f myenv.yml
```

pip install

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package-name
pip install labelme -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Reference

### RGB

- **[DDRNet](https://github.com/ydhongHIT/DDRNet)**
- **[EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch)**
- **[UNet](https://github.com/milesial/Pytorch-UNet)**
- **[TransUNet](https://github.com/Beckschen/TransUNet)**
- **[UCTransNet](https://github.com/mcgregorwwww/uctransnet)**
- **[UDTransNet](https://github.com/McGregorWwww/UDTransNet)**
- **[DIS](https://github.com/xuebinqin/DIS)**
- **[RepNeXt](https://github.com/suous/RepNeXt)**
- **[MobileOne(official)](https://github.com/apple/ml-mobileone)**
- **[MobileOne(unofficial)](https://github.com/shoutOutYangJie/MobileOne)**
- **[SegNeXt(official)](https://github.com/Visual-Attention-Network/SegNeXt)**
- **[SegNeXt(unofficial)](https://github.com/Mr-TalhaIlyas/SegNext)**
- **[fastvit(official)](https://github.com/apple/ml-fastvit)**
- **[RepLKNet(official)](https://github.com/DingXiaoH/RepLKNet-pytorch)**

### RGB-D

- **[RedNet](https://github.com/JindongJiang/RedNet)**
- **[DDRNet](https://github.com/ydhongHIT/DDRNet)**
- **[ShapeConv](https://github.com/hanchaoleng/ShapeConv)**
- **[ESANet](https://github.com/TUI-NICR/ESANet)**
- **[3D-SIS](https://github.com/Sekunde/3D-SIS)**
- **[SGNet](https://github.com/LinZhuoChen/SGNet)**
- **[CalibNet](https://github.com/PJLallen/CalibNet)**

### Paper with code

- **[NYU Depth v2](https://paperswithcode.com/sota/semantic-segmentation-on-nyu-depth-v2?tag_filter=0)**
- **[Cityscapes](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes?tag_filter=0)**

### Tools

- **[gradio](https://github.com/gradio-app/gradio)**
- **[torch-cam](https://github.com/frgfm/torch-cam)**
- **[torch-scan](https://github.com/frgfm/torch-scan)**
- **[Holocron](https://github.com/frgfm/Holocron)**
- **[detectron2](https://github.com/facebookresearch/detectron2)**

### Loss
- **[从loss处理图像分割中类别极度不均衡的状况---keras](https://blog.csdn.net/m0_37477175/article/details/83004746)**
- **[LovaszSoftmax](https://github.com/bermanmaxim/LovaszSoftmax)**
- **[pytorch-loss](https://github.com/CoinCheung/pytorch-loss)**

### Post-processing
- **[superpixPool](https://github.com/bermanmaxim/superpixPool)**
