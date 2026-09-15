import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image_path = r"D:\Code\2D-Image-Segmentation\data\oct_dataset\train\image\01_frame_000021.png"

# =========================
# 参数
# =========================

THRESHOLD = 1

# 核大小
KERNEL_SIZE = 5

# =========================
# 参数
# =========================

gray = cv2.imread(
    image_path,
    cv2.IMREAD_GRAYSCALE
)

if gray is None:
    raise FileNotFoundError(image_path)

height, width = gray.shape

# 图像中心
cx = (width - 1) / 2
cy = (height - 1) / 2

print(f"图像中心: ({cx:.1f}, {cy:.1f})")

# ============================================================
# 从中心向外寻找内圆边界
# ============================================================

yy, xx = np.indices(gray.shape)

distance = np.sqrt(
    (xx - cx) ** 2 +
    (yy - cy) ** 2
)

# 中心黑色区域
black_region = gray < THRESHOLD

# 只看距离中心 50~300 像素
search_region = (
        (distance > 50) &
        (distance < 300)
)

# 黑色区域
mask = black_region & search_region

# ============================================================
# 找中心黑色区域的连通域
# ============================================================

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    black_region.astype(np.uint8),
    connectivity=8
)

# 找中心所在的连通区域
center_x_int = int(round(cx))
center_y_int = int(round(cy))

center_label = labels[
    center_y_int,
    center_x_int
]

if center_label == 0:
    raise RuntimeError("没有找到中心黑色区域")

# 中心黑色区域
inner_mask = labels == center_label

# ============================================================
# 根据中心黑色区域计算半径
# ============================================================

ys, xs = np.where(inner_mask)

distances = np.sqrt(
    (xs - cx) ** 2 +
    (ys - cy) ** 2
)

# 最大距离就是内圆半径的近似值
inner_radius = distances.max()

print(f"中心黑色区域面积: {len(xs)} px")
print(f"内圆半径: {inner_radius:.2f} px")
print(f"内圆直径: {inner_radius * 2:.2f} px")

# ============================================================
# 可视化
# ============================================================

fig, ax = plt.subplots(figsize=(8, 8))

ax.imshow(gray, cmap="gray")

# 内圆
circle = plt.Circle(
    (cx, cy),
    inner_radius,
    fill=False,
    linewidth=1,
    color="red"
)

ax.add_patch(circle)

# 圆心
ax.plot(
    cx,
    cy,
    marker="+",
    markersize=12,
    markeredgewidth=1,
    color="yellow"
)

ax.set_title(
    f"Detected Inner Circle\n"
    f"Radius = {inner_radius:.2f} px"
)

ax.set_aspect("equal")

plt.show()
