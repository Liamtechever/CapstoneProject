import os
import subprocess
import sys
from pathlib import Path


def test_text_imports_do_not_load_media_or_prompt_for_setup(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = f"""
import builtins, sys
sys.path.insert(0, {str(root)!r})
def forbidden(*args, **kwargs):
    raise AssertionError("Unexpected interactive setup")
builtins.input = forbidden
import Model, Classifier, ModelPicker, Settings, Vision, main
assert not ({{"whisper", "torch", "TTS", "sounddevice", "cv2"}} & set(sys.modules))
assert Settings.load_settings() == Settings.Settings()
"""
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env["CAPSTONE_SETTINGS_FILE"] = str(tmp_path / "missing.json")
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())
