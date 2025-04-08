import base64
import openai
import os


from Model import Model, LocalModel, _available_models, OnlineModel, deepseek_r1_1_5, gpt4_vision, o3mini, gptturbo
from Settings import format_settings_for_model_picker, load_settings
from SpeechToText import transcribe_audio
from AudioRecording import record_audio
from Classifier import classify_prompt
from ModelPicker import pick_model
from TextToSpeech import play_tts
from Take_Picture import capture_image
from dotenv import load_dotenv
from Classifier import Classification
from AudioRecording import record_audio
from SpeechToText import transcribe_audio

if __name__ == '__main__':
    # Initialize settings if it doesn't exist
    format_settings_for_model_picker()
    # Load user settings
    user_settings = load_settings()

    test_chat = o3mini.chat("How can I do this homework")
    print(test_chat)

    play_tts("How can I help you today?", speaker="p251", speed=1, pitch=0)

    record_audio()
    _prompt = transcribe_audio("tmp/response.wav")
    print(f"Prompt: {_prompt}")

    # Classify the prompt
    prompt_classification = classify_prompt(prompt=_prompt)
    print("Using this classification: ", prompt_classification)

    if prompt_classification.requires_vision:
        # Function to encode the image
        capture_image()


        # Now you can proceed with the OpenAI API client initialization
        client = openai.OpenAI()
        print(client)


        load_dotenv("secrets.env")

        openai.api_key = os.getenv("OPENAI_API_KEY")




        def encode_image(image_path):
            """Encodes an image to base64 format."""
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")


        # Prepare the image
        image_path = 'C:/Users/liamt/PycharmProjects/CapstoneProject/captured_image.jpg'
        base64_image = encode_image(image_path)

        # Call the vision model
        response = client.chat.completions.create(
            model="gpt-4o",  # Ensure this matches your vision model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ],
                }
            ],
            max_tokens=300
        )

        print(response.choices[0].message.content)


        vision_model = pick_model(available_models=_available_models, prompt_classification=prompt_classification)[0]



    else:
        # Pick the best models
        best_models = pick_model(available_models=_available_models, prompt_classification=prompt_classification)
        base_model = deepseek_r1_1_5
        model = best_models[0]


        response = base_model.chat(prompt=_prompt)

        play_tts(model.chat(prompt=("Below is the original prompt" + _prompt + "Below is the old models response" + response), system_message="YOU ARE AN AI THAT ANALYZES OUTPUTS FROM OTHER AI. Only output the improved response. Keep the response to a maximum of three sentences"), speaker="p251", speed=1, pitch=0)
        base_response = base_model.chat(prompt=_prompt)
        # print(model.chat(prompt=("Below is the original prompt" + _prompt + "Below is the old models response" + response), system_message="YOU ARE AN AI THAT ANALYZES OUTPUTS FROM OTHER AI. Only output the improved response. Keep the response to a maximum of three sentences"))


