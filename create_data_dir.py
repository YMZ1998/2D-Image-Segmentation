import os

from scripts.data_tools.common import remove_and_create_dir

if __name__ == "__main__":
    remove_and_create_dir(os.path.join("data", "oct_augmented_dataset"))
