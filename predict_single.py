import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from parse_args import get_latest_weight_path, get_model


CLASS_NAMES = ("background", "plaque", "Stent", "Calcification")
CLASS_COLORS = np.asarray([(0, 0, 0), (255, 0, 0), (0, 120, 255), (0, 255, 0)], dtype=np.uint8)


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict and display one grayscale OCT image.")
    parser.add_argument("image", type=Path, help="source image path")
    parser.add_argument("--weights", type=Path, help="checkpoint; default is the latest model")
    parser.add_argument("--arch", "-a", default="efficientnet_b1")
    parser.add_argument("--image-size", dest="image_size", type=int, default=704)
    parser.add_argument("--num-classes", dest="num_classes", type=int, default=4)
    parser.add_argument("--in-channels", dest="in_channels", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deep-supervision", dest="deep_supervision", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--roi-radius-ratio", type=float, default=0.475)
    parser.add_argument("--keep-border-info", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("predictions"))
    parser.add_argument("--no-show", action="store_true")
    # These are used only by get_model's configuration summary.
    parser.set_defaults(epochs=0, batch_size=1)
    return parser.parse_args()


def clean_circular_roi(gray: np.ndarray, radius_ratio: float) -> np.ndarray:
    if not 0 < radius_ratio <= 0.5:
        raise ValueError("--roi-radius-ratio must be in (0, 0.5]")
    height, width = gray.shape
    yy, xx = np.ogrid[:height, :width]
    radius = min(height, width) * radius_ratio
    roi = (xx - (width - 1) / 2) ** 2 + (yy - (height - 1) / 2) ** 2 <= radius**2
    cleaned = gray.copy()
    cleaned[~roi] = 0
    return cleaned


def load_weights(model: torch.nn.Module, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state = {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}
    model.load_state_dict(state)


def main() -> None:
    args = parse_cli()
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be between 0 and 1")

    device_name = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    weights_path = args.weights or Path(get_latest_weight_path(args))
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights not found: {weights_path}; train first or use --weights")

    model = get_model(args, pretrain_backbone=False)
    load_weights(model, weights_path)
    model.to(device).eval()

    source = Image.open(args.image).convert("L")
    gray = np.asarray(source)
    if not args.keep_border_info:
        gray = clean_circular_roi(gray, args.roi_radius_ratio)
    cleaned_source = Image.fromarray(gray)
    resized = cleaned_source.resize((args.image_size, args.image_size), Image.Resampling.LANCZOS)
    array = np.asarray(resized, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)

    start = time.perf_counter()
    with torch.inference_mode():
        output = model(tensor)
        logits = output["out"] if isinstance(output, dict) else output
        prediction = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000

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
    outputs = {
        "mask": args.output_dir / f"{args.image.stem}_mask.png",
        "color": args.output_dir / f"{args.image.stem}_color.png",
        "overlay": args.output_dir / f"{args.image.stem}_overlay.png",
    }
    Image.fromarray(prediction).save(outputs["mask"])
    Image.fromarray(color_mask).save(outputs["color"])
    Image.fromarray(overlay).save(outputs["overlay"])

    present = [CLASS_NAMES[index] for index in np.unique(prediction) if index != 0]
    print(f"device: {device} | inference: {elapsed_ms:.2f} ms")
    print(f"weights: {weights_path}")
    print(f"predicted classes: {', '.join(present) if present else 'background only'}")
    for name, path in outputs.items():
        print(f"{name}: {path}")

    if not args.no_show:
        figure, axes = plt.subplots(1, 3, figsize=(16, 6))
        axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Input (grayscale)")
        axes[1].imshow(color_mask)
        axes[1].set_title("Prediction")
        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay (alpha={args.alpha:.2f})")
        for axis in axes:
            axis.axis("off")
        figure.suptitle(f"{args.image.name} | {elapsed_ms:.1f} ms")
        figure.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
