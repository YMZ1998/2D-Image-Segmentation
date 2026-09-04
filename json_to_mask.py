import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from segmentation_config import LABEL_TO_MASK_VALUE


def convert(json_path: Path, output_path: Path) -> None:
    with json_path.open("r", encoding="utf-8") as file:
        annotation = json.load(file)

    width = int(annotation["imageWidth"])
    height = int(annotation["imageHeight"])
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # Higher-valued classes remain visible when annotated regions overlap.
    shapes = sorted(
        annotation.get("shapes", []),
        key=lambda shape: LABEL_TO_MASK_VALUE.get(shape.get("label", ""), 0),
    )
    for shape in shapes:
        label = shape.get("label")
        if label not in LABEL_TO_MASK_VALUE:
            raise ValueError(f"Unknown label {label!r} in {json_path}")
        if shape.get("shape_type", "polygon") != "polygon":
            raise ValueError(f"Unsupported shape type in {json_path}: {shape.get('shape_type')}")

        points = [(round(x), round(y)) for x, y in shape["points"]]
        draw.polygon(points, fill=LABEL_TO_MASK_VALUE[label])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LabelMe JSON polygons to class-index masks.")
    parser.add_argument("input_dir", type=Path, nargs="?", default=Path("data/images"))
    parser.add_argument("output_dir", type=Path, nargs="?", default=Path("data/masks"))
    args = parser.parse_args()

    json_paths = sorted(args.input_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No JSON files found in {args.input_dir}")

    for json_path in json_paths:
        output_path = args.output_dir / f"{json_path.stem}.png"
        convert(json_path, output_path)
        print(f"{json_path.name} -> {output_path}")

    mapping = ", ".join(
        ["background=0", *(f"{name}={value}" for name, value in LABEL_TO_MASK_VALUE.items())]
    )
    print(f"Converted {len(json_paths)} masks. Pixel values: {mapping}")


if __name__ == "__main__":
    main()
