import PIL.Image
import cv2
import numpy as np

COLORS = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
          (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
          (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
          (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
          (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
          (0, 64, 128), (128, 64, 12)]


def preprocessing(image, image_size):
    image = image.resize((image_size, image_size),
                         PIL.Image.BILINEAR)
    image = np.array(image, np.float32)
    image = image / 127.5 - 1
    return image


def pre_process(image_path, mask_path, image_size):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    image = np.array(image, np.float32)
    image = image / 127.5 - 1
    image = image[..., np.newaxis]
    # image = image / 255.0

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Unable to read mask: {mask_path}")
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    # Unified classes: background=0, plaque=1, Stent=2, Calcification=3.
    raw_mask = mask.copy()
    value_to_class = {0: 0, 1: 1, 127: 1, 2: 2, 192: 2, 3: 3, 244: 3, 255: 3}
    unknown_values = set(np.unique(raw_mask).tolist()) - set(value_to_class)
    if unknown_values:
        raise ValueError(f"Unknown mask pixel values in {mask_path}: {sorted(unknown_values)}")
    mask = np.zeros_like(raw_mask, dtype=np.int64)
    for value, class_id in value_to_class.items():
        mask[raw_mask == value] = class_id

    return image, mask
