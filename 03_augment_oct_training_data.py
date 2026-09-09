"""Augment only the training split of an OCT segmentation dataset."""

import argparse
import json
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from scripts.data_tools.augment_with_imgaug import build_augmenter, circular_roi, clear_outside_roi
from scripts.data_tools.common import remove_and_create_dir
from segmentation_config import MASK_VALUE_TO_CLASS_ID


def paired_paths(root: Path, split: str):
    image_dir = root / split / "image"
    mask_dir = root / split / "mask"
    images = sorted(image_dir.glob("*.png"))
    masks = {path.name: path for path in mask_dir.glob("*.png")}
    if set(path.name for path in images) != set(masks):
        raise ValueError(f"Image/mask mismatch in {root / split}")
    return [(image, masks[image.name]) for image in images]


def write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, array)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/oct_dataset"))
    parser.add_argument("--output", type=Path, default=Path("data/oct_augmented_dataset"))
    parser.add_argument("--count", type=int, default=5, help="Variants generated per training image")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if args.count < 1:
        parser.error("--count must be at least 1")
    if not source.is_dir():
        parser.error(f"Input dataset does not exist: {source}")
    if output.exists():
        remove_and_create_dir(output)

    train_pairs = paired_paths(source, "train")
    test_pairs = paired_paths(source, "test")
    np.random.seed(args.seed)
    augmenter = build_augmenter()
    valid_values = set(MASK_VALUE_TO_CLASS_ID)
    manifest = {
        "source": str(source),
        "count_per_training_image": args.count,
        "seed": args.seed,
        "original_train_samples": len(train_pairs),
        "test_samples": len(test_pairs),
        "training_samples": [],
    }
    try:
        for index, (image_path, mask_path) in enumerate(train_pairs, 1):
            image = imageio.imread(image_path)
            mask = imageio.imread(mask_path)
            if image.ndim != 2 or mask.ndim != 2 or image.shape != mask.shape:
                raise ValueError(f"Expected matching grayscale image/mask: {image_path}")
            values = set(np.unique(mask).tolist())
            if not values.issubset(valid_values):
                raise ValueError(f"Unknown mask values {sorted(values - valid_values)}: {mask_path}")
            roi = circular_roi(*image.shape, radius_ratio=0.475)
            image, mask = clear_outside_roi(image, mask, roi)
            write_png(output / "train" / "image" / image_path.name, image)
            write_png(output / "train" / "mask" / mask_path.name, mask)
            names = [image_path.name]
            segmentation = SegmentationMapsOnImage(mask.astype(np.int32), shape=image.shape)
            for variant in range(1, args.count + 1):
                deterministic = augmenter.to_deterministic()
                aug_image, aug_segmentation = deterministic(
                    image=image, segmentation_maps=segmentation
                )
                aug_mask = aug_segmentation.get_arr().astype(np.uint8)
                aug_image, aug_mask = clear_outside_roi(aug_image, aug_mask, roi)
                augmented_values = set(np.unique(aug_mask).tolist())
                if not augmented_values.issubset(values | {0}):
                    raise RuntimeError(
                        f"Augmentation introduced mask values in {mask_path}: "
                        f"{sorted(augmented_values - values - {0})}"
                    )
                name = f"{image_path.stem}_aug{variant:02d}.png"
                write_png(output / "train" / "image" / name, aug_image)
                write_png(output / "train" / "mask" / name, aug_mask)
                names.append(name)
            manifest["training_samples"].append(
                {"source": image_path.name, "outputs": names}
            )
            print(f"[{index}/{len(train_pairs)}] {image_path.name}: {len(names)} files")

        for image_path, mask_path in test_pairs:
            image_output = output / "test" / "image" / image_path.name
            mask_output = output / "test" / "mask" / mask_path.name
            image_output.parent.mkdir(parents=True, exist_ok=True)
            mask_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, image_output)
            shutil.copy2(mask_path, mask_output)

        (output / "augmentation_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise

    train_total = len(train_pairs) * (args.count + 1)
    print(f"Generated train={train_total}, test={len(test_pairs)} at {output}")


if __name__ == "__main__":
    main()
