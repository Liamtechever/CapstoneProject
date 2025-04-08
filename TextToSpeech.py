import torch
import librosa
import sounddevice as sd
import numpy as np
from TTS.api import TTS

# Select device (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load VITS (VCTK) locally for multi-speaker support
tts = TTS("tts_models/en/vctk/vits").to(device)


import numpy as np
import sounddevice as sd

# def play_tts(text, speaker="p251", speed=1, pitch=0):
#     wav = some_tts_function(text, speaker, speed, pitch)  # Your TTS model output
#
#     # 🔹 Ensure `wav` is a NumPy array
#     wav = np.array(wav, dtype=np.float32)  # Convert list to NumPy array
#
#     if len(wav.shape) == 1:  # Mono audio
#         wav = np.expand_dims(wav, axis=0)
#
#     sd.play(wav, samplerate=22050)
#     sd.wait()

def play_tts(text, speaker="p227", speed=1.0, pitch=-1):
    print(f"Generating speech... Speaker: {speaker}, Speed: {speed}, Pitch: {pitch}")

    # Generate speech with VITS
    wav = tts.tts(text=text, speaker=speaker, speed=speed)

    # Convert to NumPy array
    wav_np = np.array(wav, dtype=np.float32)

    # Apply pitch shift using librosa
    wav_shifted = librosa.effects.pitch_shift(wav_np, sr=22050, n_steps=pitch)

    # Play the modified audio
    sd.play(wav_shifted, samplerate=22050)
    sd.wait()

   # play_tts("speak_text", speaker="p251", speed=1, pitch=0)
# Test robotic deep voice
# List of all VCTK speakers



