"""Compatibility entry point for bounded microphone recording."""

from media.audio import record_audio

__all__ = ["record_audio"]
if __name__ == "__main__":
    record_audio()
