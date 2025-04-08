import base64
import openai
import os


from Model import Model, LocalModel, _available_models, OnlineModel, deepseek_r1_1_5, gpt4_vision, o3mini, gptturbo
from ModelPicker import pick_model
from Take_Picture import capture_image
from dotenv import load_dotenv




# Function to encode the image
def UseVision(prompt, prompt_classification):

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
                    {"type": "text", "text": prompt},
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


    # vision_model = pick_model(available_models=_available_models, prompt_classification=prompt_classification)[0]