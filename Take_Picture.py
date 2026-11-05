"""Compatibility entry point for shared webcam capture."""

from media.camera import capture_image

__all__ = ["capture_image"]
if __name__ == "__main__":
    capture_image()
