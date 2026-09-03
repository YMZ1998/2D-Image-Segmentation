import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image


AUGMENTATION_SUFFIX = re.compile(r"_aug\d+$")


def source_id(path: Path) -> str:
    """Return the original-sample ID shared by an image and all its variants."""
    return AUGMENTATION_SUFFIX.sub("", path.stem)


def split_source_ids(source_ids: list[str], test_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    """Split by source and dataset group, preventing augmentation leakage."""
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")

    strata: dict[str, list[str]] = defaultdict(list)
    for item in sorted(set(source_ids)):
        group = item.split("_", 1)[0]
        strata[group].append(item)

    rng = random.Random(seed)
    train_ids: set[str] = set()
    test_ids: set[str] = set()
    for items in strata.values():
        rng.shuffle(items)
        if len(items) == 1:
            test_count = 0
        else:
            test_count = min(len(items) - 1, max(1, round(len(items) * test_ratio)))
        test_ids.update(items[:test_count])
        train_ids.update(items[test_count:])
    return train_ids, test_ids


def prepare_dataset(
    input_root: Path = Path("data/augmented"),
    output_root: Path = Path("data/dataset"),
    image_size: int = 704,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    input_images = input_root / "images"
    input_masks = input_root / "masks"
    image_paths = sorted(input_images.glob("*.png"))
    pairs = [(path, input_masks / path.name) for path in image_paths if (input_masks / path.name).exists()]
    if not pairs:
        raise FileNotFoundError(f"No paired PNG files found in {input_root}")
    if image_size < 1:
        raise ValueError("image_size must be positive")

    train_ids, test_ids = split_source_ids([source_id(path) for path, _ in pairs], test_ratio, seed)
    expected: set[Path] = set()
    counts = {"train": 0, "test": 0}

    for image_path, mask_path in pairs:
        split = "test" if source_id(image_path) in test_ids else "train"
        image_output = output_root / split / "image" / image_path.name
        mask_output = output_root / split / "mask" / mask_path.name
        image_output.parent.mkdir(parents=True, exist_ok=True)
        mask_output.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(image_path) as image:
            image.convert("L").resize((image_size, image_size), Image.Resampling.LANCZOS).save(image_output)
        with Image.open(mask_path) as mask:
            mask.convert("L").resize((image_size, image_size), Image.Resampling.NEAREST).save(mask_output)
        expected.update((image_output, mask_output))
        counts[split] += 1

    # Remove only stale generated PNGs from the four known output directories.
    for split in ("train", "test"):
        for kind in ("image", "mask"):
            directory = output_root / split / kind
            directory.mkdir(parents=True, exist_ok=True)
            for path in directory.glob("*.png"):
                if path not in expected:
                    path.unlink()

    manifest = {
        "input_root": str(input_root),
        "image_size": image_size,
        "test_ratio": test_ratio,
        "seed": seed,
        "train_source_ids": sorted(train_ids),
        "test_source_ids": sorted(test_ids),
        "train_samples": counts["train"],
        "test_samples": counts["test"],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Prepared {counts['train']} train and {counts['test']} test pairs at "
        f"{image_size}x{image_size} in {output_root}"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Group-split and resize an augmented segmentation dataset.")
    parser.add_argument("--input-root", type=Path, default=Path("data/augmented"))
    parser.add_argument("--output-root", type=Path, default=Path("data/dataset"))
    parser.add_argument("--image-size", type=int, default=704)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_dataset(args.input_root, args.output_root, args.image_size, args.test_ratio, args.seed)


if __name__ == "__main__":
    main()
