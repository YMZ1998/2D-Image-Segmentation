"""Recursively list MP4 paths and save them as UTF-8 text."""

import argparse
import os
from pathlib import Path

import cv2

import json

DEFAULT_ROOT = Path(r"D:\Filez\李进\DownLoad\12、影像资料\美国临床：支架内再狭窄")


def find_mp4_files(root: Path) -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"目录不存在或不是文件夹：{root}")

    def raise_walk_error(error: OSError) -> None:
        raise error

    paths = []
    for directory, _, filenames in os.walk(root, onerror=raise_walk_error):
        for name in filenames:
            if Path(name).suffix.lower() == ".mp4":
                paths.append(Path(directory) / name)
    return sorted(paths, key=lambda path: str(path).casefold())


def main() -> None:
    parser = argparse.ArgumentParser(description="递归查找目录下的全部 MP4，保存完整路径列表。")
    parser.add_argument("directory", type=Path, nargs="?", default=DEFAULT_ROOT, help="待遍历目录")
    parser.add_argument("-o", "--output", type=Path, default=Path("mp4_paths.txt"),
                        help="输出文本路径（默认：mp4_paths.txt）")
    args = parser.parse_args()
    try:
        paths = find_mp4_files(args.directory)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(f"{path}\n" for path in paths), encoding="utf-8")
    except OSError as error:
        parser.exit(1, f"错误：{error}\n")

    for path in paths:
        print(path)
    print(f"\n共找到 {len(paths)} 个 MP4 文件，列表已保存到：{args.output.resolve()}")
    rows = [];
    total = 0;
    errors = []
    for p in paths:
        c = cv2.VideoCapture(str(p))
        if not c.isOpened(): errors.append(str(p)); c.release(); continue
        n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = c.get(cv2.CAP_PROP_FPS);
        w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
        c.release()
        total += n
        rows.append(
            {'file': str(p.relative_to(DEFAULT_ROOT)), 'frames': n, 'size': str(w) + 'x' + str(h), 'fps': round(fps, 3),
             'seconds': round(n / fps, 2) if fps else None})
    print(json.dumps({'videos': rows, 'total_frames': total, 'errors': errors}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
