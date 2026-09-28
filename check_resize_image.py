import cv2
import numpy as np

# 输入图片路径
image_path = r'input.png'

# 读取 1024 图像
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise RuntimeError(f"无法读取图片: {image_path}")

print(f"原始尺寸: {image.shape[1]} x {image.shape[0]}")

# 1024 -> 512
image_512 = cv2.resize(
    image,
    (512, 512),
    interpolation=cv2.INTER_AREA
)

# 512 -> 1024，仅用于显示对比
image_512_up = cv2.resize(
    image_512,
    (1024, 1024),
    interpolation=cv2.INTER_LINEAR
)

# 拼接显示
comparison = np.hstack([
    image,
    image_512_up
])

cv2.imshow("1024 vs 512", comparison)

print("左边：原始 1024")
print("右边：512 缩放后再放大到 1024")
print("按任意键退出")

cv2.waitKey(0)
cv2.destroyAllWindows()