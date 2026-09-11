# 2D-Image-Segmentation

OCT single-channel semantic segmentation for background, plaque, Stent, and InvalidRegion.

## Repository layout

```text
2D-Image-Segmentation/
├── apps/                 # Qt image, mask, prediction, and MP4 viewers
├── docs/                 # Training guide and generated file lists
├── network/              # Segmentation model definitions
├── scripts/
│   ├── data_tools/       # Extraction, mask conversion, split, and augmentation
│   ├── inference/        # Single-image PyTorch and ONNX inference
│   └── development/      # Diagnostics and experimental utilities
├── utils/                # Dataset, preprocessing, loss, and evaluation modules
├── 01_extract_annotated_images.py
├── 02_prepare_oct_training_data.py
├── 03_augment_oct_training_data.py
├── 04_convert_onnx.py
├── train.py              # Training entry point
├── test.py               # Evaluation entry point
├── parse_args.py         # Shared CLI and model construction
├── segmentation_config.py
└── inference_utils.py
```

Detailed training instructions are in [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md).

## Usage

export environment

```
conda env list
conda create -n AI python=3.10 -y
conda activate AI
conda env export -n AI > myenv.yml
python -m pip freeze > requirements.txt
```

create environment

```
conda env create -f myenv.yml
```
```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

pip install

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package-name

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pillow
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple onnxruntime
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple imageio
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple imgaug 
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyqt5 

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple labelme

python -m pip install torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Data pipeline and training

The training pipeline uses single-channel images and four classes: background,
plaque, Stent, and InvalidRegion. Run the numbered data pipeline from the repository root:

```powershell
python 01_extract_annotated_images.py
python 02_prepare_oct_training_data.py
python 03_augment_oct_training_data.py
```

Train and evaluate the generated `data/oct_augmented_dataset`:

```powershell
python train.py --arch unet --batch_size 4
python test.py --arch unet --batch_size 4
```

Useful viewers and inference commands:

```powershell
python apps/inference_viewer.py
python apps/overlay_viewer.py
python apps/mp4_viewer.py
python scripts/inference/predict_single.py path/to/image.png --arch unet
python scripts/inference/predict_single_onnx.py path/to/image.png
python 04_convert_onnx.py --arch unet
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
