from pathlib import Path
from typing import Optional, Union

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_image_files(image_dir: Union[str, Path]) -> list[Path]:
    directory = Path(image_dir)
    if not directory.exists():
        return []

    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )


def find_image_for_stem(image_dir: Union[str, Path], stem: str) -> Optional[Path]:
    for path in list_image_files(image_dir):
        if path.stem == stem:
            return path
    return None
