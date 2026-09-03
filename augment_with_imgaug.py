import argparse
from pathlib import Path

import imageio.v2 as imageio
import imgaug.augmenters as iaa
import numpy as np
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def build_augmenter() -> iaa.Augmenter:
    """Build transformations safe for paired image/segmentation augmentation."""
    return iaa.Sequential(
        [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.2),
            # The following intensity transforms affect only the source image;
            # imgaug leaves SegmentationMapsOnImage class IDs unchanged.
            iaa.SomeOf(
                (0, 3),
                [
                    iaa.GaussianBlur(sigma=(0.0, 1.2)),
                    iaa.AdditiveGaussianNoise(scale=(0, 0.025 * 255)),
                    iaa.LinearContrast((0.80, 1.25)),
                    iaa.Multiply((0.85, 1.15)),
                ],
                random_order=True,
            ),
        ],
        random_order=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment paired segmentation images and masks with imgaug.")
    parser.add_argument("--input-root", type=Path, default=Path("data/merged"))
    parser.add_argument("--output-root", type=Path, default=Path("data/augmented"))
    parser.add_argument("--count", type=int, default=5, help="Augmented variants per source image")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--roi-radius-ratio",
        type=float,
        default=0.475,
        help="Circular valid-field radius divided by the shorter image side (default: 0.475)",
    )
    parser.add_argument(
        "--keep-border-info",
        action="store_true",
        help="Keep text/logo pixels outside the circular OCT field (not recommended)",
    )
    parser.add_argument(
        "--exclude-originals",
        action="store_true",
        help="Do not copy original image/mask pairs into the output dataset",
    )
    return parser.parse_args()


def circular_roi(height: int, width: int, radius_ratio: float) -> np.ndarray:
    if not 0 < radius_ratio <= 0.5:
        raise ValueError("--roi-radius-ratio must be in the interval (0, 0.5]")
    center_x = (width - 1) / 2
    center_y = (height - 1) / 2
    radius = min(height, width) * radius_ratio
    yy, xx = np.ogrid[:height, :width]
    return (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2


def clear_outside_roi(image: np.ndarray, mask: np.ndarray, roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cleaned_image = image.copy()
    cleaned_mask = mask.copy()
    cleaned_image[~roi] = 0
    cleaned_mask[~roi] = 0
    return cleaned_image, cleaned_mask


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be at least 1")

    input_images = args.input_root / "images"
    input_masks = args.input_root / "masks"
    output_images = args.output_root / "images"
    output_masks = args.output_root / "masks"
    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_images.glob("*.png"))
    pairs = [(path, input_masks / path.name) for path in image_paths if (input_masks / path.name).exists()]
    if not pairs:
        raise FileNotFoundError(f"No paired PNG images and masks found under {args.input_root}")

    np.random.seed(args.seed)
    augmenter = build_augmenter()
    expected_outputs: set[Path] = set()

    for pair_index, (image_path, mask_path) in enumerate(pairs):
        image = imageio.imread(image_path)
        mask = imageio.imread(mask_path)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Size mismatch: {image_path} {image.shape[:2]} vs {mask_path} {mask.shape[:2]}")

        roi = circular_roi(image.shape[0], image.shape[1], args.roi_radius_ratio)
        if not args.keep_border_info:
            image, mask = clear_outside_roi(image, mask, roi)

        original_values = set(np.unique(mask).tolist())
        if not args.exclude_originals:
            original_image_output = output_images / image_path.name
            original_mask_output = output_masks / mask_path.name
            imageio.imwrite(original_image_output, image)
            imageio.imwrite(original_mask_output, mask)
            expected_outputs.update((original_image_output, original_mask_output))

        segmentation = SegmentationMapsOnImage(mask.astype(np.int32), shape=image.shape)
        for variant in range(1, args.count + 1):
            # A deterministic copy guarantees identical spatial transforms for
            # the image and its segmentation map.
            deterministic = augmenter.to_deterministic()
            augmented_image, augmented_segmentation = deterministic(
                image=image,
                segmentation_maps=segmentation,
            )
            augmented_mask = augmented_segmentation.get_arr().astype(mask.dtype)
            if not args.keep_border_info:
                augmented_image, augmented_mask = clear_outside_roi(augmented_image, augmented_mask, roi)
            augmented_values = set(np.unique(augmented_mask).tolist())
            if not augmented_values.issubset(original_values | {0}):
                raise RuntimeError(
                    f"Unexpected mask values for {mask_path.name}: {sorted(augmented_values)}"
                )

            output_name = f"{image_path.stem}_aug{variant:02d}.png"
            image_output = output_images / output_name
            mask_output = output_masks / output_name
            imageio.imwrite(image_output, augmented_image)
            imageio.imwrite(mask_output, augmented_mask)
            expected_outputs.update((image_output, mask_output))

        print(f"[{pair_index + 1}/{len(pairs)}] {image_path.name}: generated {args.count} variants")

    # Keep reruns reproducible when --count or input data changes.
    for directory in (output_images, output_masks):
        for path in directory.glob("*.png"):
            if path not in expected_outputs:
                path.unlink()

    total = len(pairs) * args.count + (0 if args.exclude_originals else len(pairs))
    print(f"Done: {len(pairs)} sources -> {total} paired samples in {args.output_root}")


if __name__ == "__main__":
    main()
