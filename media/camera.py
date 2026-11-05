"""Shared webcam capture with explicit success and device cleanup."""

import time
from pathlib import Path

from core.errors import MediaError


def capture_image(save_path="captured_image.jpg"):
    try:
        import cv2
    except ImportError as exc:
        raise MediaError("Install the optional camera dependencies first.") from exc
    cap = cv2.VideoCapture(0)
    try:
        if not cap.isOpened():
            raise MediaError("The webcam could not be opened.")
        time.sleep(0.5)
        ok, frame = cap.read()
        if not ok:
            raise MediaError("The webcam did not return an image.")
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), frame):
            raise MediaError("The captured image could not be saved.")
        return str(path)
    except MediaError:
        raise
    except Exception as exc:
        raise MediaError(
            "Camera capture failed. Check device access and the save location."
        ) from exc
    finally:
        cap.release()
