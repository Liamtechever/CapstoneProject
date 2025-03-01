import whisper
from AudioRecording import record_audio


def transcribe_audio(file_path: str, model: str = "small"):
    model = whisper.load_model(model)
    result = model.transcribe(file_path)

    return result["text"]



if __name__ == '__main__':

    record_audio()
    print(transcribe_audio("output.mp3"))


