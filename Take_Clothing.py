"""Capture a wardrobe image with a collision-resistant filename."""

from uuid import uuid4

from core.config import PROJECT_ROOT
from media.camera import capture_image


def TakePicture():
    return capture_image(PROJECT_ROOT / "var" / "wardrobe" / f"{uuid4().hex}.jpg")


if __name__ == "__main__":
    TakePicture()
