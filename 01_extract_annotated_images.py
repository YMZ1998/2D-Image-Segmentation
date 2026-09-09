"""Copy LabelMe-annotated images and JSON files, preserving subdirectories."""

import argparse
import json
import shutil
from pathlib import Path

from scripts.data_tools.common import remove_and_create_dir

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_pairs(source: Path):
    pairs = []
    for annotation in sorted(source.rglob("*.json")):
        data = json.loads(annotation.read_text(encoding="utf-8-sig"))
        # Skip export manifests and other JSON documents. Empty shapes are valid
        # annotations for normal images with no target objects.
        if not isinstance(data, dict) or not isinstance(data.get("shapes"), list):
            continue
        candidates = [p for p in annotation.parent.iterdir()
                      if p.is_file() and p.stem == annotation.stem
                      and p.suffix.lower() in IMAGE_SUFFIXES]
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one matching image: {annotation}; found {len(candidates)}")
        pairs.append((candidates[0], annotation))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(r"D:\data\OCT"))
    parser.add_argument("--output", type=Path, default=Path("Extracted"))
    parser.add_argument("--dry-run", action="store_true", help="Check and count files without copying")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not source.is_dir():
        parser.error(f"Source directory does not exist: {source}")
    if output == source or source in output.parents:
        parser.error("Output must be outside the source directory")
    remove_and_create_dir(args.output)
    pairs = collect_pairs(source)
    transfers = [(path, output / path.relative_to(source)) for pair in pairs for path in pair]
    # Validate all conflicts before copying; repeat runs may reuse identical files.
    pending = []
    for src, dst in transfers:
        if dst.exists():
            if not dst.is_file() or src.read_bytes() != dst.read_bytes():
                raise FileExistsError(f"Different file already exists; refusing overwrite: {dst}")
        else:
            pending.append((src, dst))
    print(f"Annotated image/JSON pairs: {len(pairs)}")
    print(f"Files to copy: {len(pending)}; identical files skipped: {len(transfers) - len(pending)}")
    print(f"Output: {output}")
    if args.dry_run:
        return
    for src, dst in pending:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print("Copy complete.")


if __name__ == "__main__":
    main()
