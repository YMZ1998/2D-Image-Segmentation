import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from PIL import Image


CLASS_NAMES = ("background", "plaque", "Stent", "Calcification")
CLASS_COLORS = np.asarray([(0, 0, 0), (255, 0, 0), (0, 120, 255), (0, 255, 0)], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image segmentation with the latest ONNX model.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, help="ONNX path; defaults to newest save_weights/*.onnx")
    parser.add_argument("--output-dir", type=Path, default=Path("predictions_onnx"))
    parser.add_argument("--image-size", type=int, default=704, help="fallback for dynamic ONNX dimensions")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--roi-radius-ratio", type=float, default=0.475)
    parser.add_argument("--keep-border-info", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="force CPU inference")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def newest_onnx() -> Path:
    candidates = list(Path("save_weights").glob("*.onnx"))
    if not candidates:
        raise FileNotFoundError("No ONNX model found in save_weights/*.onnx")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def clean_circular_roi(gray: np.ndarray, radius_ratio: float) -> np.ndarray:
    if not 0 < radius_ratio <= 0.5:
        raise ValueError("--roi-radius-ratio must be in (0, 0.5]")
    height, width = gray.shape
    yy, xx = np.ogrid[:height, :width]
    radius = min(height, width) * radius_ratio
    roi = (xx - (width - 1) / 2) ** 2 + (yy - (height - 1) / 2) ** 2 <= radius**2
    result = gray.copy()
    result[~roi] = 0
    return result


def fixed_dimension(value, fallback: int) -> int:
    return int(value) if isinstance(value, int) and value > 0 else fallback


def prepare_input(gray: np.ndarray, shape: list, fallback_size: int) -> tuple[np.ndarray, tuple[int, int], str]:
    if len(shape) != 4:
        raise ValueError(f"Expected a 4D ONNX input, got: {shape}")
    if shape[1] in (1, 3):
        layout, channels = "NCHW", int(shape[1])
        height, width = fixed_dimension(shape[2], fallback_size), fixed_dimension(shape[3], fallback_size)
    elif shape[3] in (1, 3):
        layout, channels = "NHWC", int(shape[3])
        height, width = fixed_dimension(shape[1], fallback_size), fixed_dimension(shape[2], fallback_size)
    else:
        raise ValueError(f"Cannot determine channel/layout from ONNX input shape: {shape}")

    resized = np.asarray(Image.fromarray(gray).resize((width, height), Image.Resampling.LANCZOS), dtype=np.float32)
    resized = resized / 127.5 - 1.0
    if channels == 3:
        resized = np.repeat(resized[..., None], 3, axis=2)
    else:
        resized = resized[..., None]
    tensor = resized.transpose(2, 0, 1)[None] if layout == "NCHW" else resized[None]
    return np.ascontiguousarray(tensor, dtype=np.float32), (height, width), layout


def logits_to_mask(output: np.ndarray) -> np.ndarray:
    output = np.asarray(output)
    if output.ndim == 4:
        if output.shape[1] == len(CLASS_NAMES):
            return output.argmax(axis=1)[0].astype(np.uint8)
        if output.shape[-1] == len(CLASS_NAMES):
            return output.argmax(axis=-1)[0].astype(np.uint8)
        raise ValueError(f"Cannot find the class axis in ONNX output shape: {output.shape}")
    if output.ndim == 3 and output.shape[0] == 1:
        return output[0].astype(np.uint8)
    if output.ndim == 2:
        return output.astype(np.uint8)
    raise ValueError(f"Unsupported ONNX output shape: {output.shape}")


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be between 0 and 1")

    model_path = args.model or newest_onnx()
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    providers = ["CPUExecutionProvider"]
    if not args.cpu and "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_meta = session.get_inputs()[0]

    source = Image.open(args.image).convert("L")
    gray = np.asarray(source)
    if not args.keep_border_info:
        gray = clean_circular_roi(gray, args.roi_radius_ratio)
    tensor, inference_size, layout = prepare_input(gray, input_meta.shape, args.image_size)

    start = time.perf_counter()
    outputs = session.run(None, {input_meta.name: tensor})
    elapsed_ms = (time.perf_counter() - start) * 1000
    prediction = logits_to_mask(outputs[0])
    prediction = np.asarray(Image.fromarray(prediction).resize(source.size, Image.Resampling.NEAREST))
    if int(prediction.max()) >= len(CLASS_COLORS):
        raise ValueError(f"Invalid predicted class ID: {prediction.max()}")

    color_mask = CLASS_COLORS[prediction]
    gray_rgb = np.repeat(gray[..., None], 3, axis=2)
    overlay = gray_rgb.copy()
    foreground = prediction != 0
    overlay[foreground] = (
        (1 - args.alpha) * gray_rgb[foreground] + args.alpha * color_mask[foreground]
    ).astype(np.uint8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "mask": args.output_dir / f"{args.image.stem}_mask.png",
        "color": args.output_dir / f"{args.image.stem}_color.png",
        "overlay": args.output_dir / f"{args.image.stem}_overlay.png",
    }
    Image.fromarray(prediction).save(paths["mask"])
    Image.fromarray(color_mask).save(paths["color"])
    Image.fromarray(overlay).save(paths["overlay"])

    present = [CLASS_NAMES[index] for index in np.unique(prediction) if index != 0]
    print(f"model: {model_path}")
    print(f"provider: {session.get_providers()[0]}")
    print(f"input: {input_meta.name} {input_meta.shape} ({layout}, inference={inference_size})")
    print(f"inference: {elapsed_ms:.2f} ms")
    print(f"predicted classes: {', '.join(present) if present else 'background only'}")
    for name, path in paths.items():
        print(f"{name}: {path}")

    if not args.no_show:
        figure, axes = plt.subplots(1, 3, figsize=(16, 6))
        axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Input (grayscale)")
        axes[1].imshow(color_mask)
        axes[1].set_title("ONNX prediction")
        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay (alpha={args.alpha:.2f})")
        for axis in axes:
            axis.axis("off")
        figure.suptitle(f"{model_path.name} | {elapsed_ms:.1f} ms")
        figure.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
