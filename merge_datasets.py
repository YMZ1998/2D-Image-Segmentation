import shutil
from pathlib import Path


DATASETS = (
    ("group1", Path("data/images"), Path("data/masks")),
    ("group2", Path("data/image2"), Path("data/masks2")),
)
OUTPUT_ROOT = Path("data/merged")


def main() -> None:
    output_images = OUTPUT_ROOT / "images"
    output_masks = OUTPUT_ROOT / "masks"
    output_json = OUTPUT_ROOT / "json"
    for directory in (output_images, output_masks, output_json):
        directory.mkdir(parents=True, exist_ok=True)

    expected_outputs: set[Path] = set()
    sample_count = 0
    for prefix, image_dir, mask_dir in DATASETS:
        for mask_path in sorted(mask_dir.glob("*.png")):
            image_path = image_dir / mask_path.name
            if not image_path.exists():
                print(f"Skipping {prefix}/{mask_path.name}: image not found")
                continue

            output_name = f"{prefix}_{mask_path.name}"
            image_output = output_images / output_name
            mask_output = output_masks / output_name
            shutil.copy2(image_path, image_output)
            shutil.copy2(mask_path, mask_output)
            expected_outputs.update((image_output, mask_output))

            json_path = image_dir / f"{mask_path.stem}.json"
            if json_path.exists():
                json_output = output_json / f"{prefix}_{json_path.name}"
                shutil.copy2(json_path, json_output)
                expected_outputs.add(json_output)

            print(f"{prefix}/{mask_path.name} -> {output_name}")
            sample_count += 1

    # Remove stale files from earlier merge runs while keeping the operation
    # strictly scoped to the three generated output directories.
    for directory in (output_images, output_masks, output_json):
        for path in directory.iterdir():
            if path.is_file() and path not in expected_outputs:
                path.unlink()

    print(f"Merged {sample_count} paired samples into {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
