import time
import sounddevice as sd
import soundfile as sf
import numpy as np

def record_audio(filename="tmp/response.wav", samplerate=44100, channels=1, silence_threshold=0.001,
                 short_pause_duration=1.5, long_silence_duration=2.5):
    """
    Records audio and stops automatically after detecting a long silence **following** speech.

    - Allows **short pauses** in speech (up to `short_pause_duration` sec).
    - Stops recording **immediately** after a long silence (`long_silence_duration`).
    - Ensures the audio stream properly closes after stopping.

    Parameters:
    - filename (str): Name of the output file.
    - samplerate (int): Sample rate of recording.
    - channels (int): Number of audio channels.
    - silence_threshold (float): Volume level below which sound is considered silence.
    - short_pause_duration (float): Max duration of short pauses allowed.
    - long_silence_duration (float): Duration required to detect a long silence and stop.

    Returns:
    - Saves the recorded file and exits.
    """
    recording = []
    volume_history = []
    speech_detected = False
    silence_start = None
    debug_count = 0
    stop_recording = False  #  Flag to force stop recording

    def callback(indata, frames, callback_time, status):
        """Callback function to collect recorded audio and monitor silence."""
        nonlocal silence_start, speech_detected, debug_count, stop_recording

        if status:
            print(f" Audio Input Error: {status}")

        recording.append(indata.copy())

        # Measure volume level (Root Mean Square)
        volume_norm = np.linalg.norm(indata) / len(indata)

        # Keep track of recent volume levels (moving average)
        volume_history.append(volume_norm)
        if len(volume_history) > 10:  # Keep last 10 readings
            volume_history.pop(0)

        avg_volume = np.mean(volume_history)  # Smooth volume reading

        # Print live volume level (DEBUGGING)
        debug_count += 1
        if debug_count % 5 == 0:  # Print every 5 iterations to reduce spam
            print(f"🔊 Volume Level: {avg_volume:.5f} (Threshold: {silence_threshold})", end="\r")

        # If user speaks, mark speech as detected
        if avg_volume > silence_threshold:
            if not speech_detected:
                print("\n🗣️ Speech detected! Recording in progress...")
            speech_detected = True
            silence_start = None  # Reset silence timer when speech is detected
        elif speech_detected:  # Track silence *only after* speech starts
            if silence_start is None:
                silence_start = time.monotonic()  # ✅ FIXED: Use monotonic() to track silence duration
                print("\n Silence detected, waiting to confirm...")
            else:
                elapsed_silence = time.monotonic() - silence_start  # TODO: Maybe don't have weird multiple types with this variable

                # 🔹 **NEW: Allow short pauses**
                if elapsed_silence < short_pause_duration:
                    print(f"⏳ Short pause: {elapsed_silence:.2f} sec", end="\r")
                elif elapsed_silence >= long_silence_duration:
                    print(f"\n Long silence detected ({elapsed_silence:.2f}s), stopping recording.")
                    stop_recording = True  #  Set flag to force stop
                    raise sd.CallbackAbort  #  Stop recording immediately

    print("\n🎤 Recording... Speak when ready.")

    try:
        with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback) as stream:
            while not stop_recording:  # ✅ Only runs while recording is active
                time.sleep(0.1)  # Prevent CPU overload
    except sd.CallbackAbort:
        print("\n Stopping stream...")

    # Ensure the recording contains valid audio
    if len(recording) > 0:
        audio_data = np.concatenate(recording, axis=0)
        sf.write(filename, audio_data, samplerate)
        print(f" Saved recording as {filename}")
    else:
        print(" No audio detected. Try speaking louder.")

    print(" Moving to transcription step...")  # Debugging - Ensure it moves forward
