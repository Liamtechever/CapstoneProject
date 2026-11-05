import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.errors import MediaError
from media.audio import SilenceTracker
from media.camera import capture_image
from media.speech import _whisper_model, transcribe_audio


def test_initial_silence_times_out():
    tracker = SilenceTracker(0)
    assert not tracker.update(9, False)
    assert tracker.update(10, False)
    assert not tracker.heard_speech


def test_short_pause_is_allowed_and_long_pause_stops():
    tracker = SilenceTracker(0)
    assert not tracker.update(1, True)
    assert not tracker.update(2, False)
    assert not tracker.update(3, False)
    assert not tracker.update(4, True)
    assert not tracker.update(5, False)
    assert tracker.update(7.5, False)


def test_continuous_speech_has_maximum_duration():
    tracker = SilenceTracker(0)
    assert tracker.update(60, True)


def test_transcription_reuses_model(monkeypatch):
    model = SimpleNamespace(transcribe=Mock(return_value={"text": " words "}))
    loader = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "whisper", SimpleNamespace(load_model=loader))
    _whisper_model.cache_clear()
    try:
        assert transcribe_audio("first.wav") == "words"
        assert transcribe_audio("second.wav") == "words"
        loader.assert_called_once_with("small")
    finally:
        _whisper_model.cache_clear()


@pytest.mark.parametrize(
    "opened,frame_ok,write_ok", [(False, False, False), (True, False, False), (True, True, False)]
)
def test_camera_failure_releases_device(tmp_path, monkeypatch, opened, frame_ok, write_ok):
    import media.camera as camera

    cap = Mock()
    cap.isOpened.return_value = opened
    cap.read.return_value = (frame_ok, "frame")
    cv2 = SimpleNamespace(VideoCapture=Mock(return_value=cap), imwrite=Mock(return_value=write_ok))
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setattr(camera.time, "sleep", lambda _: None)
    with pytest.raises(MediaError):
        capture_image(tmp_path / "test.jpg")
    cap.release.assert_called_once()


def test_camera_returns_saved_path(tmp_path, monkeypatch):
    import media.camera as camera

    cap = Mock()
    cap.isOpened.return_value = True
    cap.read.return_value = (True, "frame")
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(VideoCapture=Mock(return_value=cap), imwrite=Mock(return_value=True)),
    )
    monkeypatch.setattr(camera.time, "sleep", lambda _: None)
    path = tmp_path / "test.jpg"
    assert capture_image(path) == str(path)
    cap.release.assert_called_once()
