import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from PIL import Image

from inference_utils import (
    clean_circular_roi,
    colorize_mask,
    newest_onnx,
    onnx_output_to_mask,
    overlay_prediction,
    prepare_onnx_input,
)
from segmentation_config import CLASS_NAMES, IMAGE_SIZE, ROI_RADIUS_RATIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image segmentation with the latest ONNX model.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, help="ONNX path; defaults to newest save_weights/*.onnx")
    parser.add_argument("--output-dir", type=Path, default=Path("predictions_onnx"))
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE, help="fallback for dynamic ONNX dimensions")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--roi-radius-ratio", type=float, default=ROI_RADIUS_RATIO)
    parser.add_argument("--keep-border-info", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="force CPU inference")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


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
    tensor, inference_size, layout = prepare_onnx_input(gray, input_meta.shape, args.image_size)

    start = time.perf_counter()
    outputs = session.run(None, {input_meta.name: tensor})
    elapsed_ms = (time.perf_counter() - start) * 1000
    prediction = onnx_output_to_mask(outputs[0])
    prediction = np.asarray(Image.fromarray(prediction).resize(source.size, Image.Resampling.NEAREST))
    color_mask = colorize_mask(prediction)
    gray_rgb = np.repeat(gray[..., None], 3, axis=2)
    overlay = overlay_prediction(gray_rgb, prediction, args.alpha)

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
