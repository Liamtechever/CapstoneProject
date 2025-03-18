import re
import json
from dataclasses import dataclass

from Model import Model, LocalModel, _available_models, OnlineModel
from Settings import format_settings_for_modelpicker, load_settings
from SpeechToText import transcribe_audio
from AudioRecording import record_audio
from Classifier import classify_prompt
from ModelPicker import pick_model
from TextToSpeech import play_tts
from Classifier import Classification

if __name__ == '__main__':
    # Initialize settings if it doesn't exist
    format_settings_for_modelpicker()

    play_tts("How can I help you today?", speaker="p251", speed=1, pitch=0)

    record_audio()
    _prompt = transcribe_audio("response.wav")
    print(f"Prompt: {_prompt}")

    # Classify the prompt
    prompt_classification = classify_prompt(prompt=_prompt)
    print(prompt_classification)

    # Load user settings
    user_settings = load_settings()

    # Pick the best models
    best_models = pick_model(available_models=_available_models, prompt_classification=prompt_classification)

    print(best_models)
