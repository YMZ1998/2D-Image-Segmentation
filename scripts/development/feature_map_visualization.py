import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import torch

from parse_args import parse_args, get_model, get_best_weight_path, get_device


def mask_postprocessing(mask, w, h):
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask[mask == 1] = 64
    mask[mask == 2] = 129
    mask[mask == 3] = 192
    mask[mask == 4] = 255
    return mask


args = parse_args()
device = get_device()

# create model
model = get_model(args)
# load weights
weights_path = get_best_weight_path(args)
model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
model.to(device)
model.eval()

# 加载和预处理输入图像
img_path = 'test.png'  # 替换为你自己的图像路径
original_img = cv2.imread(img_path)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

image = cv2.resize(original_img, (args.image_size, args.image_size),
                   interpolation=cv2.INTER_LINEAR)
image = image / 127.5 - 1
image = torch.Tensor(image)
image = image.permute(2, 0, 1)
image = torch.unsqueeze(image, dim=0).to(device)

# 选择你感兴趣的中间层
layer_name = 'backbone.4'  # 替换为你感兴趣的层名称
activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


print(dict([*model.named_modules()]).keys())
# 注册钩子
layer = dict([*model.named_modules()])[layer_name]
layer.register_forward_hook(get_activation(layer_name))

# 获取中间层输出的特征图
output = model(image)
act = activation[layer_name].squeeze()


# 可视化特征图
def plot_feature_maps(feature_maps, col_size=4, row_size=4):
    fig, ax = plt.subplots(row_size, col_size, figsize=(12, 12))
    for i in range(row_size):
        for j in range(col_size):
            index = i * col_size + j
            if index < feature_maps.shape[0]:
                ax[i][j].imshow(feature_maps[index].cpu().numpy(), cmap='viridis')
            ax[i][j].axis('off')
    plt.show()


# 绘制特征图
num_features = min(act.shape[0], 16)  # 选择前64个特征图进行可视化
plot_feature_maps(act[:num_features], col_size=4, row_size=4)
