import sounddevice as sd
import numpy as np
import ffmpeg
import soundfile as sf



def record_audio(filename="output.mp3", samplerate=44100, channels=2):
    """Records audio until Enter is pressed and saves it as an MP3 file."""

    recording = []  # Stores recorded audio

    def callback(indata, frames, time, status):
        """Callback function to collect recorded audio."""
        if status:
            print(status)
        recording.append(indata.copy())

    print("Recording... Press Enter to stop.")

    with sd.InputStream(samplerate=samplerate, channels=channels, dtype=np.int16, callback=callback):
        input()  # Wait for user to press Enter

    print("Recording stopped.")

    # Convert list of NumPy arrays to a single array
    audio_data = np.concatenate(recording, axis=0)

    # Save as WAV first (needed for MP3 conversion)
    wav_filename = filename.replace(".mp3", ".wav")
    sf.write(wav_filename, audio_data, samplerate)

    # Convert to MP3 using ffmpeg
    ffmpeg.input(wav_filename).output(filename, format="mp3").run(overwrite_output=True)

    print(f"Saved as {filename}")


# Example Usage:
if __name__ == "__main__":
    record_audio("test.mp3")
