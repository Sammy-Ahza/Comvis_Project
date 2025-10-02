import os
from typing import List

HAM_CLASSES: List[str] = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]


def resolve_image_path(data_root: str, image_id: str) -> str:
    """Resolve path to image by checking known HAM10000 folders."""
    candidates = [
        os.path.join(data_root, "ham10000_images_part_1", f"{image_id}.jpg"),
        os.path.join(data_root, "ham10000_images_part_2", f"{image_id}.jpg"),
        os.path.join(data_root, "HAM10000_images_part_1", f"{image_id}.jpg"),
        os.path.join(data_root, "HAM10000_images_part_2", f"{image_id}.jpg"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Image not found for id={image_id} in {data_root}")
