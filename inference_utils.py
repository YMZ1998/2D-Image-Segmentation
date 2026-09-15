"""Shared image and ONNX inference helpers used by CLI and Qt tools."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from segmentation_config import (
    CLASS_COLORS,
    CLASS_NAMES,
    IMAGE_SIZE,
    ROI_RADIUS_RATIO,
    ROI_INNER_RADIUS_RATIO, OUTER_RADIUS, INNER_RADIUS,
)


def newest_onnx(directory: Path = Path("save_weights")) -> Path:
    candidates = list(directory.glob("*.onnx"))
    if not candidates:
        raise FileNotFoundError(f"No ONNX model found in {directory}/*.onnx")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def circular_roi_mask(
    height: int,
    width: int,
    radius_ratio: float = ROI_RADIUS_RATIO,
    inner_radius_ratio: float = ROI_INNER_RADIUS_RATIO,
) -> np.ndarray:
    if not 0 < radius_ratio <= 0.5:
        raise ValueError("radius_ratio must be in (0, 0.5]")
    if not 0 <= inner_radius_ratio < radius_ratio:
        raise ValueError("inner_radius_ratio must be in [0, radius_ratio)")

    yy, xx = np.ogrid[:height, :width]
    center_x = (width - 1) / 2
    center_y = (height - 1) / 2
    distance_sq = (xx - center_x) ** 2 + (yy - center_y) ** 2
    outer = min(height, width) * radius_ratio
    inner = min(height, width) * inner_radius_ratio
    return (distance_sq <= outer**2) & (distance_sq >= inner**2)


def clean_circular_roi(
    array: np.ndarray,
    radius_ratio: float = ROI_RADIUS_RATIO,
    inner_radius_ratio: float = ROI_INNER_RADIUS_RATIO,
    fill_value: int = 0,
) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale array, got shape {array.shape}")

    roi = circular_roi_mask(
        array.shape[0],
        array.shape[1],
        radius_ratio=radius_ratio,
        inner_radius_ratio=inner_radius_ratio,
    )
    cleaned = array.copy()
    cleaned[~roi] = fill_value
    return cleaned


def create_pseudocolor(gray: np.ndarray) -> np.ndarray:
    gray = np.asarray(ImageOps.autocontrast(Image.fromarray(gray), cutoff=0.5))
    anchors = [
        (0, (0, 0, 0)),
        (45, (22, 4, 1)),
        (100, (76, 15, 3)),
        (160, (163, 55, 7)),
        (215, (245, 139, 24)),
        (255, (255, 232, 135)),
    ]
    lookup = np.zeros((256, 3), dtype=np.uint8)
    for start, end in zip(anchors[:-1], anchors[1:]):
        values = np.arange(start[0], end[0] + 1)
        ratio = (values - start[0]) / (end[0] - start[0])
        lookup[values] = np.asarray(start[1]) + ratio[:, None] * (
                np.asarray(end[1]) - np.asarray(start[1])
        )
    return lookup[gray]


def _fixed_dimension(value, fallback: int) -> int:
    return int(value) if isinstance(value, int) and value > 0 else fallback


def prepare_onnx_input(
        gray: np.ndarray, shape: list, fallback_size: int = IMAGE_SIZE
) -> tuple[np.ndarray, tuple[int, int], str]:
    if len(shape) != 4:
        raise ValueError(f"Expected a 4D ONNX input, got: {shape}")
    if shape[1] in (1, 3):
        layout, channels = "NCHW", int(shape[1])
        height = _fixed_dimension(shape[2], fallback_size)
        width = _fixed_dimension(shape[3], fallback_size)
    elif shape[3] in (1, 3):
        layout, channels = "NHWC", int(shape[3])
        height = _fixed_dimension(shape[1], fallback_size)
        width = _fixed_dimension(shape[2], fallback_size)
    else:
        raise ValueError(f"Cannot determine channel/layout from ONNX input shape: {shape}")

    resized = np.asarray(
        Image.fromarray(gray).resize((width, height), Image.Resampling.LANCZOS), dtype=np.float32
    )
    resized = resized / 127.5 - 1.0
    resized = np.repeat(resized[..., None], channels, axis=2)
    tensor = resized.transpose(2, 0, 1)[None] if layout == "NCHW" else resized[None]
    return np.ascontiguousarray(tensor, dtype=np.float32), (height, width), layout


def onnx_output_to_mask(output: np.ndarray) -> np.ndarray:
    output = np.asarray(output)
    if output.ndim == 4:
        # Accept the previous four-class model as well as the current five-class model.
        valid_class_counts = range(2, len(CLASS_NAMES) + 1)
        if output.shape[1] in valid_class_counts:
            return output.argmax(axis=1)[0].astype(np.uint8)
        if output.shape[-1] in valid_class_counts:
            return output.argmax(axis=-1)[0].astype(np.uint8)
        raise ValueError(f"Cannot find the class axis in ONNX output shape: {output.shape}")
    if output.ndim == 3 and output.shape[0] == 1:
        return output[0].astype(np.uint8)
    if output.ndim == 2:
        return output.astype(np.uint8)
    raise ValueError(f"Unsupported ONNX output shape: {output.shape}")


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    if mask.size and int(mask.max()) >= len(CLASS_COLORS):
        raise ValueError(f"Invalid predicted class ID: {mask.max()}")
    return np.asarray(CLASS_COLORS, dtype=np.uint8)[mask]


def overlay_prediction(base_rgb: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    colors = colorize_mask(mask)
    result = base_rgb.copy()
    foreground = mask != 0
    result[foreground] = (
            (1 - alpha) * result[foreground] + alpha * colors[foreground]
    ).astype(np.uint8)
    return result


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from matplotlib.patches import Circle

    image_path = r"D:\Code\2D-Image-Segmentation\data\oct_dataset\train\image\01_frame_000021.png"

    gray = np.array(Image.open(image_path).convert("L"))
    cleaned = clean_circular_roi(gray)

    cx = cy = 512
    # =========================
    # 可视化
    # =========================
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 原图 + 圆环
    axes[0].imshow(gray, cmap="gray")

    axes[0].add_patch(
        Circle(
            (cx, cy),
            INNER_RADIUS,
            fill=False,
            linewidth=1,
            edgecolor="blue",
        )
    )

    axes[0].add_patch(
        Circle(
            (cx, cy),
            OUTER_RADIUS,
            fill=False,
            linewidth=1,
            edgecolor="red",
        )
    )

    axes[0].scatter(
        cx,
        cy,
        s=50,
        c="yellow",
        marker="+",
    )

    axes[0].set_title(
        f"Original\n"
        f"Inner R={INNER_RADIUS}px, "
        f"Outer R={OUTER_RADIUS}px"
    )

    axes[0].set_aspect("equal")

    # 圆环 ROI 后的结果
    axes[1].imshow(cleaned, cmap="gray")
    axes[1].set_title("Circular Ring ROI")
    axes[1].set_aspect("equal")

    plt.tight_layout()
    plt.show()
