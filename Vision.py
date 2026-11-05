"""Compatibility entry point for the explicitly requested camera workflow."""

from pathlib import Path
from tempfile import TemporaryDirectory

from core.service import generate_response
from media.camera import capture_image


def UseVision(prompt, prompt_classification=None):
    with TemporaryDirectory() as folder:
        image = capture_image(Path(folder) / "capture.jpg")
        response, _ = generate_response(prompt, image=image)
        return response
