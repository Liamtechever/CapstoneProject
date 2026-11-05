# Optional speech, camera, and wardrobe experiments

The text app does not require these packages. The media dependency set is separate because it includes large ML libraries and platform-specific components.

The maintained [coqui-tts package](https://coqui-tts.readthedocs.io/en/latest/installation.html) preserves the `TTS.api` interface used by the project. It replaces the original unmaintained `TTS` distribution; do not install both into one environment. Model choices remain Whisper `small` and Coqui `tts_models/en/vctk/vits`. This is a library migration, not a new trained model.

Use a fresh Python 3.12 environment. Install FFmpeg and an audio backend for your OS; VITS may also require eSpeak NG. PyTorch/torchaudio/torchcodec must match the platform and one another. Review [Coqui's installation notes](https://coqui-tts.readthedocs.io/en/latest/installation.html) before choosing a CUDA/CPU build.

```sh
python -m pip install --require-hashes -r requirements-media.txt
python -m pip check
python main.py --voice
python main.py --camera "Describe what is visible"
```

The media lock is generated for Python 3.12 on Windows. It is not a universal GPU lock. On another platform or for a different PyTorch wheel source, regenerate from `requirements-media.in` in an isolated environment and audit the result before use. Do not mix mismatched torch and torchaudio versions.

The first explicit transcription/speech call may download pretrained weights to library caches. Importing the application does not download them. Respect the upstream model licenses.

## Behavior

- Recording stops after 10 seconds without speech, 2.5 seconds of silence after speech, or a 60-second maximum.
- Whisper is cached by model name for repeated calls.
- TTS uses the synthesizer's output sample rate, with optional pitch adjustment.
- Camera failures return a controlled error and release the device.
- CLI recordings/images are removed when their temporary directory closes.
- `python Settings.py` starts optional voice setup explicitly, with three attempts per answer.
- `python Fashion.py` opens the independent wardrobe program. It retains images/database under ignored `var/wardrobe/`. Internet mode and sufficient cost tolerance must be enabled for hosted image descriptions or outfit suggestions; manual descriptions remain available.

## Manual acceptance checks

Use the intended hardware to check silence-only recording, continuous speech timeout, transcription accuracy, playback speed/pitch, unavailable devices, camera save failure, local/hosted vision, and wardrobe capture/display/deletion. Automated tests use fake devices and providers; they do not certify these hardware/model paths.

Do not claim that the wardrobe model verifies clothing categories: IDs, duplicates, and unknown selections are checked, while category interpretation remains model-generated.
