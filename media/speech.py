"""Lazy, cached speech models. No downloads occur until a speech function is called."""

from functools import lru_cache

from core.errors import MediaError


@lru_cache(maxsize=2)
def _whisper_model(name):
    import whisper

    return whisper.load_model(name)


def transcribe_audio(file_path, model="small"):
    try:
        result = _whisper_model(model).transcribe(str(file_path))
        text = result["text"].strip()
        if not text:
            raise MediaError("No speech could be transcribed.")
        return text
    except MediaError:
        raise
    except Exception as exc:
        raise MediaError(
            "Transcription failed. Check the media installation, model cache, and FFmpeg."
        ) from exc


@lru_cache(maxsize=1)
def _tts_model():
    import torch
    from TTS.api import TTS

    return TTS("tts_models/en/vctk/vits").to("cuda" if torch.cuda.is_available() else "cpu")


def play_tts(text, speaker="p251", speed=1.0, pitch=0):
    if not text or not 0.5 <= speed <= 2 or not -12 <= pitch <= 12:
        raise MediaError("Invalid speech text, speed, or pitch.")
    try:
        import librosa
        import numpy as np
        import sounddevice as sd

        tts = _tts_model()
        waveform = np.asarray(tts.tts(text=text, speaker=speaker, speed=speed), dtype=np.float32)
        sample_rate = tts.synthesizer.output_sample_rate
        if pitch:
            waveform = librosa.effects.pitch_shift(waveform, sr=sample_rate, n_steps=pitch)
        sd.play(waveform, samplerate=sample_rate)
        sd.wait()
    except Exception as exc:
        raise MediaError(
            "Speech playback failed. Check optional media dependencies and the output device."
        ) from exc
