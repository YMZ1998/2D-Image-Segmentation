import os
import shutil
from typing import List


def remove_and_create_dir(path: str):
    """删除目录并重新创建"""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def get_base_dir(path: str) -> str:
    """返回父目录路径"""
    return os.path.dirname(path)


def get_new_path(path: str, file_name: str) -> str:
    """返回与原路径同目录下的新文件路径"""
    return os.path.join(os.path.dirname(path), file_name)


def list_files(path: str, exts: List[str] = None) -> List[str]:
    """
    列出目录下的所有文件，可指定文件后缀过滤
    exts 示例: ['.nii', '.gz']
    """
    if not os.path.isdir(path):
        return []

    files = []
    for f in os.listdir(path):
        full = os.path.join(path, f)
        if os.path.isfile(full):
            if exts is None or any(f.lower().endswith(e.lower()) for e in exts):
                files.append(full)
    return files


def copy_files(src_dir: str, dst_dir: str, exts: List[str] = None):
    """复制某目录下的文件到目标目录"""
    os.makedirs(dst_dir, exist_ok=True)

    files = list_files(src_dir, exts)
    for file in files:
        shutil.copy(file, dst_dir)
    return files


def rename_file(path: str, new_name: str) -> str:
    """重命名文件并返回新路径"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} is not a file")

    new_path = os.path.join(os.path.dirname(path), new_name)
    os.rename(path, new_path)
    return new_path


def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    input_path = r"D:\\debug\\test.nii.gz"
    print("父目录:", get_base_dir(input_path))
    print("新路径:", get_new_path(input_path, 'seg.nii.gz'))
