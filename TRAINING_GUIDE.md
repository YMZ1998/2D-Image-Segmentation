# OCT 图像分割训练说明

## 1. 任务与类别

本项目使用单通道 OCT 图像进行四类语义分割：

| 训练 ID | 类别 | 英文名称 | 原始 mask 灰度值 |
| --- | --- | --- | --- |
| 0 | 背景 | background | 0 |
| 1 | 斑块 | plaque | 127 |
| 2 | 支架 | Stent | 192 |
| 3 | 钙化 | Calcification | 244 |

`utils/preprocess.py` 在训练时将 mask 灰度值转换为连续类别 ID `0–3`。mask 缩放必须使用最近邻插值，禁止使用会产生中间灰度值的双线性或双三次插值。

## 2. 图像格式与分辨率

- 原始图像：单通道灰度图，分辨率 `1408 × 1408`。
- 默认训练分辨率：`704 × 704`。
- 可选快速实验分辨率：`352 × 352`。

当前默认使用 `704 × 704`，以保留细小支架和钙化边缘信息。其显存与计算开销约为 `352 × 352` 的四倍；显存不足时应优先减小 batch size，也可临时切换到 `352 × 352` 做快速实验。

生成默认 704 分辨率的数据：

```powershell
python prepare_training_data.py
python train.py --skip_data_prepare
```

## 3. 阳性与正常数据

训练数据不能只包含斑块、支架或钙化等阳性图像，还必须包含正常图像，否则模型容易把正常组织、导管反光或噪声误识别为病变。

正常图像的 mask 应为同尺寸的单通道全零 PNG，即全部属于 background。正常数据需遵循以下规则：

1. 图片和全零 mask 必须同名且成对存在。
2. 正常图像参与增强，但增强后的 mask 必须仍为全零。
3. 训练集和测试集都应包含正常图像。
4. 同一病例、采集序列或原图的所有增强版本只能属于一个数据子集。
5. 如果能获取患者或病例 ID，应优先按患者或病例划分，避免同一患者同时出现在训练集和测试集。

正常数据加入 `data/merged/images` 时，应同时将同名全零 mask 放入 `data/merged/masks`。

## 4. 隐私与无效区域清理

OCT 圆形有效视野外的时间、患者信息、医院信息、设备信息和 Logo 与分割任务无关，也可能造成隐私泄露和特征污染。

`augment_with_imgaug.py` 会在增强前创建居中的圆形 ROI，将圆外图像和 mask 全部置零，并在增强后再次清理圆外区域。默认圆半径为图像短边的 `0.475` 倍：

```powershell
python augment_with_imgaug.py --roi-radius-ratio 0.475
```

调整 ROI 后必须人工抽查，确认无效信息已完全移除且 OCT 有效区域没有被裁掉。

## 5. 数据增强策略

采集图像固定居中，因此不得使用会改变视野中心或整体尺度的增强。

允许的几何增强：

- 水平翻转；
- 垂直翻转；
- 旋转，当前范围为 `-15°～15°`；
- 错切，当前范围为 `-5°～5°`。

允许的灰度强度增强：

- 高斯模糊；
- 高斯噪声；
- 对比度变化；
- 整体亮度乘法和加法变化。

禁止的增强：

- 缩放；
- 平移；
- RGB 通道扰动、色相或饱和度变换；
- 对 mask 使用非最近邻插值。

增强前会先将图像转换为单通道灰度图。几何增强对图像和 mask 使用同一组随机参数；亮度、对比度、模糊和噪声只作用于图像。默认每张原图生成 5 个增强版本，并保留清理后的原图：

```powershell
python augment_with_imgaug.py
```

输出目录为 `data/augmented/images` 和 `data/augmented/masks`。

## 6. 训练集与测试集划分

`prepare_training_data.py` 默认按原始样本 ID 做 `80%/20%` 划分。同一原图及其 `_augXX` 版本会进入同一集合，防止增强版本泄漏到测试集。

```powershell
python prepare_training_data.py
```

默认输出：

```text
data/dataset/
├── split_manifest.json
├── train/
│   ├── image/
│   └── mask/
└── test/
    ├── image/
    └── mask/
```

`split_manifest.json` 保存随机种子、训练/测试原始样本 ID 和数量。当前固定随机种子为 `42`。数据较少时，单次划分的指标波动较大；当前可固定划分用于开发，数据增多后建议使用按病例分组的交叉验证，并保留独立最终测试集。

## 7. 完整训练流程

```powershell
conda activate AI

# 1. 灰度增强并清除圆外无效信息
python augment_with_imgaug.py

# 2. 可选：单独划分并生成 704×704 数据
python prepare_training_data.py

# 3. 训练；默认会自动执行第 2 步
python train.py --arch unet --batch_size 8

# 4. 使用最佳权重测试
python test.py --arch unet --batch_size 8
```

默认训练配置：

- 数据路径：`data/dataset`；
- 输入通道：`1`；
- 图像尺寸：`704 × 704`；
- 类别数：`4`；
- 默认 batch size：`4`；
- 训练/测试比例：`80%/20%`；
- 划分随机种子：`42`。

已有准备好的数据时，可跳过重新生成：

```powershell
python train.py --skip_data_prepare
```

EfficientNet 和 MobileNet 使用 RGB 预训练骨干时，会在模型内部把单通道张量复制为三通道；磁盘图片和数据加载器仍保持单通道。UNet、UDTransNet 和 ETransUNet 原生使用单通道输入。

## 8. 训练前检查

每次新增、合并或重新标注数据后，应确认：

1. 图片与 mask 文件名一一对应，尺寸一致。
2. 图像为单通道灰度图。
3. 原始 mask 只包含 `0、127、192、244`。
4. 训练 mask 能映射为 `0、1、2、3`。
5. OCT 圆外全部为零，不包含文字、Logo 或患者信息。
6. 同一原图的增强版本没有跨训练集和测试集。
7. 训练集和测试集均包含正常样本及目标类别。
8. 使用 `qt_overlay_viewer.py` 抽查图像与 mask 的叠加位置。

## 9. 评估建议

训练过程会在每个 epoch 打印训练集和验证集的 background、plaque、Stent、Calcification Dice，并将结果写入 `log/` 下的训练日志。总体 Dice 为三个前景类别 Dice 的平均值，不包含 background。

除总体 mIoU 和 Dice 外，应分别报告 plaque、Stent 和 Calcification 的指标，并在正常图像上统计误检率。背景占比很高，不应只使用像素准确率评价模型。

每次实验还应记录原始样本数、正常样本数、增强倍率、划分随机种子、输入分辨率、模型和代码版本，确保结果可以复现和正确比较。
