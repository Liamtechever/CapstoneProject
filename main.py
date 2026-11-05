"""Text CLI by default; microphone and camera require explicit flags."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from core.errors import AssistantError
from core.service import generate_response


def main():
    parser = argparse.ArgumentParser(description="Capstone model-routing assistant")
    parser.add_argument("prompt", nargs="?", help="Text question; otherwise read from the terminal")
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Record speech and speak the answer (optional media dependencies)",
    )
    parser.add_argument(
        "--camera", action="store_true", help="Capture one webcam image for this request"
    )
    args = parser.parse_args()
    try:
        with TemporaryDirectory() as folder:
            prompt = args.prompt
            if args.voice:
                from media.audio import record_audio
                from media.speech import transcribe_audio

                prompt = transcribe_audio(record_audio(Path(folder) / "request.wav"))
            if prompt is None:
                prompt = input("Your question: ")
            image = None
            if args.camera:
                from media.camera import capture_image

                image = capture_image(Path(folder) / "capture.jpg")
            response, model = generate_response(prompt, image=image)
            print(f"{model}: {response}")
            if args.voice:
                from media.speech import play_tts

                play_tts(response)
        return 0
    except (AssistantError, EOFError, KeyboardInterrupt) as exc:
        print(str(exc) or "Cancelled.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
