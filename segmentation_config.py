"""Shared dataset, class and visualization configuration."""

IMAGE_SIZE = 1024
ROI_RADIUS_RATIO = 0.475

CLASS_NAMES = ("background", "plaque", "Stent", "InvalidRegion")
CLASS_COLORS = (
    (0, 0, 0),
    (255, 0, 0),
    (0, 120, 255),
    (255, 215, 0),
)

CLASS_ID_TO_MASK_VALUE = {0: 0, 1: 64, 2: 128, 3: 192}

# Grayscale values stored in LabelMe-derived PNG masks.
LABEL_TO_MASK_VALUE = {
    "plaque": CLASS_ID_TO_MASK_VALUE[1],
    "Stent": CLASS_ID_TO_MASK_VALUE[2],
    "InvalidRegion": CLASS_ID_TO_MASK_VALUE[3],
}

# Accept both contiguous class IDs and display-scaled grayscale masks.
MASK_VALUE_TO_CLASS_ID = {
    0: 0,
    1: 1,
    64: 1,
    2: 2,
    128: 2,
    3: 3,
    192: 3,
    # Accept InvalidRegion masks generated before Calcification was removed.
    255: 3,
}

CLASS_DISPLAY_VALUES = {
    class_id: frozenset(value for value, mapped_id in MASK_VALUE_TO_CLASS_ID.items() if mapped_id == class_id)
    for class_id in range(1, len(CLASS_NAMES))
}
