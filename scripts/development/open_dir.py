import os
import platform
import subprocess


def open_directory(directory_path):
    """
    Opens the specified directory using the default file explorer.
    """
    if platform.system() == "Windows":
        os.startfile(directory_path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", directory_path])
    else:
        subprocess.Popen(["xdg-open", directory_path])


if __name__ == "__main__":
    current_directory = os.getcwd()
    print(current_directory)
    open_directory(current_directory)
