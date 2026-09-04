"""Shared dataset, class and visualization configuration."""

IMAGE_SIZE = 704
ROI_RADIUS_RATIO = 0.475

CLASS_NAMES = ("background", "plaque", "Stent", "Calcification")
CLASS_COLORS = (
    (0, 0, 0),
    (255, 0, 0),
    (0, 120, 255),
    (0, 255, 0),
)

# Grayscale values stored in LabelMe-derived PNG masks.
LABEL_TO_MASK_VALUE = {
    "plaque": 127,
    "Stent": 192,
    "Calcification": 244,
}

# Accept both contiguous class IDs and display-scaled grayscale masks.
MASK_VALUE_TO_CLASS_ID = {
    0: 0,
    1: 1,
    127: 1,
    2: 2,
    192: 2,
    3: 3,
    244: 3,
    255: 3,
}

CLASS_DISPLAY_VALUES = {
    class_id: frozenset(value for value, mapped_id in MASK_VALUE_TO_CLASS_ID.items() if mapped_id == class_id)
    for class_id in range(1, len(CLASS_NAMES))
}

