"""Build an OCT segmentation dataset without changing the source resolution."""

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from dir_process import remove_and_create_dir
from inference_utils import clean_circular_roi
from segmentation_config import CLASS_ID_TO_MASK_VALUE
# LabelMe source label -> contiguous training class ID.
SOURCE_LABEL_TO_CLASS_ID = {
    "1": 1,  # plaque
    "2": 2,  # Stent
    "3": 3,  # InvalidRegion
    "plaque": 1,
    "Stent": 2,
    "InvalidRegion": 3,
}
CLASS_NAMES = ("background", "plaque", "Stent", "InvalidRegion")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def find_image(annotation: Path) -> Path:
    matches = [annotation.with_suffix(suffix) for suffix in IMAGE_SUFFIXES]
    matches = [path for path in matches if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"Expected one image for {annotation}, found {len(matches)}")
    return matches[0]


def read_sample(annotation: Path):
    data = json.loads(annotation.read_text(encoding="utf-8-sig"))
    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError(f"Invalid LabelMe JSON: {annotation}")
    labels = [str(shape.get("label", "")) for shape in shapes]
    if not shapes:
        return None, "empty"
    if "object" in labels:
        return None, "object"
    unknown = sorted(set(labels) - set(SOURCE_LABEL_TO_CLASS_ID))
    if unknown:
        raise ValueError(f"Unknown labels {unknown} in {annotation}")
    return data, None


def draw_mask(data: dict) -> Image.Image:
    size = (int(data["imageWidth"]), int(data["imageHeight"]))
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for shape in data["shapes"]:
        if shape.get("shape_type", "polygon") != "polygon":
            raise ValueError(f"Unsupported shape type: {shape.get('shape_type')}")
        points = [(round(x), round(y)) for x, y in shape["points"]]
        class_id = SOURCE_LABEL_TO_CLASS_ID[str(shape["label"])]
        draw.polygon(points, fill=CLASS_ID_TO_MASK_VALUE[class_id])
    return mask


def choose_test_groups(groups: list[str], ratio: float, seed: int) -> set[str]:
    if not 0 <= ratio < 1:
        raise ValueError("test_ratio must be in [0, 1)")
    if len(groups) < 2 or ratio == 0:
        return set()
    shuffled = sorted(groups)
    random.Random(seed).shuffle(shuffled)
    count = min(len(groups) - 1, max(1, round(len(groups) * ratio)))
    return set(shuffled[:count])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("Extracted"))
    parser.add_argument("--output", type=Path, default=Path("data/oct_dataset"))
    parser.add_argument("--test-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        parser.error(f"Input directory does not exist: {source}")
    if output.exists():
        remove_and_create_dir(output)

    samples = []
    filtered = Counter()
    for annotation in sorted(source.rglob("*.json")):
        data, reason = read_sample(annotation)
        if reason:
            filtered[reason] += 1
            continue
        image = find_image(annotation)
        group = annotation.relative_to(source).parts[0]
        samples.append((group, image, annotation, data))
    if not samples:
        raise RuntimeError("No valid annotated samples found")

    groups = sorted({sample[0] for sample in samples})
    test_groups = choose_test_groups(groups, args.test_ratio, args.seed)
    manifest = {
        "source": str(source),
        "image_size": "original",
        "class_names": list(CLASS_NAMES),
        "source_label_to_class_id": SOURCE_LABEL_TO_CLASS_ID,
        "test_groups": sorted(test_groups),
        "filtered": dict(filtered),
        "samples": [],
    }
    try:
        for group, image_path, annotation, data in samples:
            split = "test" if group in test_groups else "train"
            name = f"{group}_{image_path.stem}.png"
            image_output = output / split / "image" / name
            mask_output = output / split / "mask" / name
            image_output.parent.mkdir(parents=True, exist_ok=True)
            mask_output.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(image_path) as image:
                gray = np.asarray(image.convert("L"))
                cleaned = Image.fromarray(clean_circular_roi(gray))
                cleaned.save(image_output)
            mask = draw_mask(data)
            if mask.size != cleaned.size:
                raise ValueError(
                    f"Image/annotation size mismatch: {image_path} {cleaned.size}, mask {mask.size}"
                )
            mask.save(mask_output)
            manifest["samples"].append({
                "name": name,
                "group": group,
                "split": split,
                "annotation": str(annotation),
                "mask_values": sorted(set(mask.getdata())),
            })
        (output / "dataset_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise

    counts = Counter(sample["split"] for sample in manifest["samples"])
    print(f"Generated {len(samples)} samples at {output}")
    print(f"train={counts['train']}, test={counts['test']}, test groups={sorted(test_groups)}")
    print(f"filtered: object={filtered['object']}, empty={filtered['empty']}")


if __name__ == "__main__":
    main()
