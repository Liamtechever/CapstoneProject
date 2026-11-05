"""Optional spoken setup, invoked only by an explicit user command."""

from core.config import Settings, save_settings
from core.errors import AssistantError


def configure():
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from media.audio import record_audio
    from media.speech import play_tts, transcribe_audio

    def ask(question, valid):
        with TemporaryDirectory() as folder:
            for _ in range(3):
                play_tts(question)
                text = (
                    transcribe_audio(record_audio(Path(folder) / "answer.wav"))
                    .lower()
                    .strip(" .!?")
                )
                if text in valid:
                    return valid[text]
            raise AssistantError(
                "Voice setup could not understand the answer after three attempts."
            )

    online = ask("Allow internet-based models? Say yes or no.", {"yes": True, "no": False})
    numbers = dict(
        zip(
            ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
            range(11),
        )
    )
    numbers.update({str(n): n for n in range(11)})
    budget = (
        ask(
            "Choose a relative cost tolerance from zero to ten. This is not a dollar limit.",
            numbers,
        )
        if online
        else 0
    )
    settings = Settings(use_internet=online, api_cost_tolerance=budget)
    save_settings(settings)
    play_tts(
        "Settings saved. The memory limit defaults to four gigabytes; edit the settings file to change it."
    )
    return settings
