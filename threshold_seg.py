import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_inner_circle(
        image_path: str,
        threshold: int = 1,
        max_radius: int = 300,
        visualize: bool = True,
):
    gray = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if gray is None:
        raise FileNotFoundError(image_path)

    height, width = gray.shape

    cx = (width - 1) / 2
    cy = (height - 1) / 2

    # 黑色区域
    black_region = (
            gray < threshold
    ).astype(np.uint8)

    # 连通域
    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            black_region,
            connectivity=8
        )
    )

    # 中心像素所在的连通区域
    center_x = int(round(cx))
    center_y = int(round(cy))

    center_label = labels[
        center_y,
        center_x
    ]

    if center_label == 0:
        raise RuntimeError(
            "图像中心不是黑色区域，请调整 threshold"
        )

    # 中心黑色区域
    inner_mask = (
            labels == center_label
    )

    # 获取坐标
    ys, xs = np.where(inner_mask)

    distances = np.sqrt(
        (xs - cx) ** 2 +
        (ys - cy) ** 2
    )

    # 只保留合理范围
    distances = distances[
        distances < max_radius
        ]

    if len(distances) == 0:
        raise RuntimeError(
            "没有找到有效的内圆区域"
        )

    # 使用百分位数比 max 更稳定
    inner_radius = np.percentile(
        distances,
        99.99
    )
    inner_radius = int(inner_radius)
    # print(f"图像中心: ({cx:.1f}, {cy:.1f})")
    # print(f"内圆半径: {inner_radius:.2f} px")
    # print(f"内圆直径: {inner_radius * 2:.2f} px")

    if visualize:
        fig, ax = plt.subplots(
            figsize=(8, 8)
        )

        ax.imshow(
            gray,
            cmap="gray"
        )

        circle = plt.Circle(
            (cx, cy),
            inner_radius,
            fill=False,
            linewidth=1,
            color="red"
        )

        ax.add_patch(circle)

        ax.plot(
            cx,
            cy,
            "+",
            markersize=12,
            color="yellow"
        )

        ax.set_title(
            f"Inner Radius = "
            f"{inner_radius:.2f}px"
        )

        ax.set_aspect("equal")

        plt.show()

    return inner_radius, (cx, cy)


if __name__ == "__main__":
    src_path = r"D:\Code\2D-Image-Segmentation\Extracted"
    for p in os.listdir(src_path):
        path = os.path.join(src_path, p)
        paths = glob.glob(os.path.join(path, "*.png"))
        image_path = os.path.join(path, paths[0])

        print(image_path)

        inner_radius, center = detect_inner_circle(
            image_path,
            threshold=1,
            max_radius=300,
            visualize=False
        )

        print("最终内圆半径:", inner_radius)
        # print("最终圆心:", center)
