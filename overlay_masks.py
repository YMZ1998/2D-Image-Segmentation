import argparse
from pathlib import Path

from PIL import Image


CLASSES = {
    # Support both class-index masks and display-scaled grayscale masks.
    "plaque": ({1, 127}, (255, 0, 0)),  #斑块
    "Calcification": ({2, 244, 255}, (0, 255, 0)), #钙化
}


def create_overlay(image_path: Path, mask_path: Path, output_path: Path, alpha: float) -> None:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if image.size != mask.size:
        raise ValueError(f"Size mismatch: {image_path} {image.size}, {mask_path} {mask.size}")

    result = image.copy()
    for class_values, color in CLASSES.values():
        # An explicit lookup table reliably preserves class IDs 1 and 2.
        lookup_table = [255 if value in class_values else 0 for value in range(256)]
        class_mask = mask.point(lookup_table)
        color_layer = Image.new("RGB", image.size, color)
        blended = Image.blend(result, color_layer, alpha)
        result.paste(blended, mask=class_mask)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay class-index masks on source images.")
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--mask-dir", type=Path, default=Path("data/masks"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/overlays"))
    parser.add_argument("--alpha", type=float, default=0.45)
    args = parser.parse_args()

    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be between 0 and 1")

    mask_paths = sorted(args.mask_dir.glob("*.png"))
    if not mask_paths:
        raise FileNotFoundError(f"No masks found in {args.mask_dir}")

    converted = 0
    for mask_path in mask_paths:
        image_path = args.image_dir / mask_path.name
        if not image_path.exists():
            print(f"Skipping {mask_path.name}: source image not found")
            continue
        output_path = args.output_dir / mask_path.name
        create_overlay(image_path, mask_path, output_path, args.alpha)
        print(f"{image_path.name} + {mask_path.name} -> {output_path}")
        converted += 1

    print(f"Created {converted} overlays. Red=plaque, green=Calcification")


if __name__ == "__main__":
    main()
