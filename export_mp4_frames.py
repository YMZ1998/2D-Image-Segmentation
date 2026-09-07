"""Export every Nth decoded video frame as a grayscale PNG."""

import argparse
import json
from pathlib import Path

import cv2

from dir_process import remove_and_create_dir
from list_mp4_files import DEFAULT_ROOT, find_mp4_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=Path(r"D:\data\OCT"))
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interrupted export; verify existing PNGs before skipping")
    args = parser.parse_args()
    if args.step < 1:
        parser.error("--step must be positive")
    videos = find_mp4_files(args.input)
    if not videos:
        parser.error("No MP4 videos found")
    remove_and_create_dir(args.output)
    folders = [args.output / f"{i:02d}" for i in range(1, len(videos) + 1)]
    manifest_path = args.output / "export_manifest.json"
    for path in [*folders, manifest_path]:
        if path.exists() and not args.resume:
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    if args.resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["input_root"] != str(args.input.resolve()) or previous["step"] != args.step:
            raise ValueError("Existing export uses a different input directory or frame step")
        for item, video in zip(previous["videos"], videos):
            if item["source"] != str(video):
                raise ValueError("Video ordering has changed; use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"input_root": str(args.input.resolve()), "step": args.step,
                "first_frame": 1, "mode": "grayscale", "videos": []}
    for video, folder in zip(videos, folders):
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot open video: {video}")
        folder.mkdir(exist_ok=args.resume)
        reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        decoded = saved = 0
        try:
            while capture.grab():
                index = decoded
                decoded += 1
                if index % args.step:
                    continue
                ok, frame = capture.retrieve()
                if not ok:
                    raise RuntimeError(f"Cannot retrieve frame {index + 1}: {video}")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ok, encoded = cv2.imencode(".png", gray, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                if not ok:
                    raise RuntimeError(f"PNG encoding failed: {video}")
                # Source frame number (one-based) makes each PNG traceable.
                destination = folder / f"frame_{index + 1:06d}.png"
                payload = encoded.tobytes()
                if args.resume and destination.exists():
                    if destination.read_bytes() != payload:
                        raise ValueError(f"Existing PNG differs from source frame: {destination}")
                else:
                    with destination.open("xb") as stream:
                        stream.write(payload)
                saved += 1
        finally:
            capture.release()
        manifest["videos"].append({"folder": folder.name, "source": str(video),
                                   "reported_frames": reported, "decoded_frames": decoded,
                                   "exported_pngs": saved})
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{folder.name}: decoded={decoded}, reported={reported}, PNGs={saved}", flush=True)
        if decoded != reported:
            print(f"WARNING: frame count differs for folder {folder.name}", flush=True)
    print(f"DONE: {sum(v['exported_pngs'] for v in manifest['videos'])} PNGs in {args.output}", flush=True)


if __name__ == "__main__":
    main()
