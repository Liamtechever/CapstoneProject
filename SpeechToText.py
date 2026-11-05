"""Compatibility entry point for cached Whisper transcription."""

from media.speech import transcribe_audio

__all__ = ["transcribe_audio"]
if __name__ == "__main__":
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from media.audio import record_audio

    with TemporaryDirectory() as folder:
        print(transcribe_audio(record_audio(Path(folder) / "response.wav")))
