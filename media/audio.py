"""Bounded microphone recording; silence detection is independently testable."""

import time
from dataclasses import dataclass
from pathlib import Path

from core.errors import MediaError


@dataclass
class SilenceTracker:
    started_at: float
    initial_timeout: float = 10
    silence_duration: float = 2.5
    max_duration: float = 60
    heard_speech: bool = False
    silence_since: float | None = None

    def update(self, now, loud):
        if now - self.started_at >= self.max_duration:
            return True
        if loud:
            self.heard_speech = True
            self.silence_since = None
        elif self.heard_speech:
            if self.silence_since is None:
                self.silence_since = now
            if now - self.silence_since >= self.silence_duration:
                return True
        return not self.heard_speech and now - self.started_at >= self.initial_timeout


def record_audio(
    filename="tmp/response.wav",
    samplerate=44100,
    channels=1,
    silence_threshold=0.01,
    short_pause_duration=1.5,
    long_silence_duration=2.5,
    initial_timeout=10,
    max_duration=60,
):
    if not (
        0 < short_pause_duration <= long_silence_duration < max_duration
        and 0 < initial_timeout <= max_duration
        and samplerate > 0
        and channels > 0
    ):
        raise MediaError("Invalid recording limits.")
    try:
        import numpy as np
        import sounddevice as sd
        import soundfile as sf
    except ImportError as exc:
        raise MediaError("Install the optional media dependencies first.") from exc
    tracker = SilenceTracker(time.monotonic(), initial_timeout, long_silence_duration, max_duration)
    blocks = []
    stopped = False
    input_failed = False

    def callback(indata, frames, callback_time, status):
        nonlocal stopped, input_failed
        if status:
            input_failed = True
            stopped = True
            raise sd.CallbackAbort
        blocks.append(indata.copy())
        rms = float(np.sqrt(np.mean(indata.astype(float) ** 2)))
        stopped = tracker.update(time.monotonic(), rms > silence_threshold)
        if stopped:
            raise sd.CallbackAbort

    try:
        with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback) as stream:
            while not stopped:
                now = time.monotonic()
                # Also terminate if the device stops delivering callbacks.
                if now - tracker.started_at >= max_duration:
                    break
                if not tracker.heard_speech and now - tracker.started_at >= initial_timeout:
                    break
                if not stream.active:
                    raise MediaError("The microphone stopped unexpectedly.")
                time.sleep(0.05)
        if input_failed:
            raise MediaError("The microphone reported an input error.")
        if not tracker.heard_speech or not blocks:
            raise MediaError("No speech detected before the recording timeout.")
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), np.concatenate(blocks, axis=0), samplerate)
        return str(path)
    except MediaError:
        raise
    except Exception as exc:
        raise MediaError("Recording failed. Check microphone access and the audio device.") from exc
