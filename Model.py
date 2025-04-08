from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from ollama import ChatResponse, chat


from dotenv import load_dotenv
import os
import openai
from sympy.physics.units import speed

from AudioRecording import record_audio
from SpeechToText import transcribe_audio

load_dotenv("secrets.env")

openai.api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key is None:
    raise ValueError("API key not found. Check your secrets.env file.")



class Strengths:
    def __init__(self, subject_strength: str, strength_level: int):
        self.subject_strength = subject_strength
        self.strength_level = strength_level


class Model(ABC):
    # TODO: Speed
    # TODO: is_reasoning_model
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    @abstractmethod
    def vram(self) -> int:
        pass

    @property
    @abstractmethod
    def strengths(self) -> List[Strengths]:
        pass

    @property
    @abstractmethod
    def response_speed(self) -> int:
        pass

    @property
    @abstractmethod
    def cost(self) -> int:
        pass

    @property
    @abstractmethod
    def vision_model(self) -> bool:
       pass

    @abstractmethod
    def chat(self, prompt: str, system_message: str = "", image: str = None):
        pass

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return self.__str__()

num = 8

text = str(num)


class LocalModel(Model):
    def __init__(self, name: str, vram: int, strengths: List[Strengths], response_speed: int, cost: int, vision_model: bool = False):
        super().__init__(name)
        self._vram = vram
        self._strengths = strengths
        self._speed = response_speed
        self._cost = cost
        self._vision_model = vision_model

    @property
    def vram(self) -> int:
        return self._vram

    @property
    def strengths(self) -> List[Strengths]:
        return self._strengths

    @property
    def response_speed(self) -> int:
        pass

    @property
    def vision_model(self) -> bool:
       return self._vision_model

    @property
    def cost(self) -> int:
        return self._cost

    def chat(self, prompt: str, system_message: str = "", image: str = None):

        if image:
            response: ChatResponse = chat(
                model=self.name,
                messages=[
                    # {
                    #     "role": "system",
                    #     "content": system_message
                    # },
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image]
                    }
                ]
            )

        else:
            response: ChatResponse = chat(
                model=self.name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"{prompt}"},
                ]
            )


        response_text = response.message.content.strip()

        return response_text


class OnlineModel(Model):
    def __init__(self, name: str, vram: int, strengths: List[Strengths], response_speed: int, cost: int, vision_model: bool = False):
        super().__init__(name)
        self._vram = vram
        self._strengths = strengths
        self._response_speed = speed
        self._cost = cost
        self._vision_model = vision_model

    @property
    def vram(self) -> int:
        return self._vram

    @property
    def strengths(self) -> List[Strengths]:
        return self._strengths

    @property
    def response_speed(self) -> int:
        pass

    @property
    def cost(self) -> int:
        return self._cost

    @property
    def vision_model(self) -> bool:
       return self._vision_model

    def chat(self, prompt: str, system_message: str = "", image: str = None):
        # Insert OpenAPI library calls here

        if image:
            response = openai.chat.completions.create(
                model=self.name,  # GPT-4 Vision model
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "user", "content": [image]}
                ]
            )


        else:
            response = openai.chat.completions.create(
                model=self.name,  # Or "gpt-3.5-turbo" if GPT-4 is unavailable
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"{prompt}"},
                ],
                # max_completion_tokens=500,

            )


        response_text = response.choices[0].message.content.strip()
        print(response.__dict__)

        return response_text



deepseek_r1_671b = LocalModel(
    name="deepseek-r1:671b",
    vram=48,  # Requires high VRAM due to large model size
    strengths=[
        Strengths("NLP", 9),
        Strengths("Coding", 9),
        Strengths("Math", 9),
        Strengths("Science", 8),
        Strengths("Technology", 9),
        Strengths("Engineering", 8),
        Strengths("Business_and_Economics", 7),
        Strengths("History", 7),
        Strengths("Literature", 6),
        Strengths("Philosophy", 7),
        Strengths("Other", 7)
    ],
    response_speed=10,
    cost=0  # Local model, no API cost
)

deepseek_r1_8b = LocalModel(
    name="deepseek-r1:8b",
    vram=16,  # Requires a moderate amount of VRAM
    strengths=[
        Strengths("NLP", 9),
        Strengths("Coding", 8),
        Strengths("Math", 8),
        Strengths("Science", 7),
        Strengths("Technology", 8),
        Strengths("Engineering", 7),
        Strengths("Business_and_Economics", 6),
        Strengths("History", 6),
        Strengths("Literature", 5),
        Strengths("Philosophy", 6),
        Strengths("Other", 6)
    ],
    response_speed=7,
    cost=0  # Local model, no API cost
)

deepseek_r1_1_5 = LocalModel(
    name="deepseek-r1:1.5b",
    vram=4,  # Lower VRAM requirement for a small model
    strengths=[
        Strengths("NLP", 9),
        Strengths("Reasoning", 8),
        Strengths("Math", 7),
        Strengths("Science", 6),
        Strengths("Technology", 7),
        Strengths("Engineering", 6),
        Strengths("Business_and_Economics", 6),
        Strengths("History", 6),
        Strengths("Literature", 6),
        Strengths("Philosophy", 8),
        Strengths("Other", 6)
    ],
    response_speed=5,
    cost=0  # Local model, no API cost
)

tinyllama = LocalModel(
    name="tinyllama",
    vram=2,  # Very lightweight, runs on minimal VRAM
    strengths=[
        Strengths("Coding", 9),
        Strengths("Technology", 7),
        Strengths("Engineering", 6),
        Strengths("Other", 5)
    ],
    response_speed=3,
    cost=0  # Local model, no API cost
)

gptturbo = OnlineModel(
    name="gpt-3.5-turbo",
    vram=0,  # Online model, no local VRAM required
    strengths=[
        Strengths("Coding", 5),
        Strengths("Technology", 5),
        Strengths("Business_and_Economics", 5),
        Strengths("History", 5),
        Strengths("Literature", 5),
        Strengths("Philosophy", 5),
        Strengths("Other", 6)
    ],
    response_speed=0,
    cost=5  # API cost: $0.50 per 1M tokens x 10
)

o3mini = OnlineModel(
    name="o3-mini",
    vram=0,  # Online model, no local VRAM required
    strengths=[
        Strengths("Coding", 5),
        Strengths("Technology", 5),
        Strengths("Business_and_Economics", 5),
        Strengths("History", 5),
        Strengths("Literature", 5),
        Strengths("Philosophy", 5),
        Strengths("Other", 6)
    ],
    response_speed=0,
    cost=3  # API cost: $0.30 per 1M tokens x 10
)

llama3_2_vision = LocalModel(
    name="llama3.2-vision",
    vram=24,  # Estimated VRAM requirement for multimodal processing
    strengths=[
        Strengths("NLP", 9),
        Strengths("Coding", 7),
        Strengths("Math", 8),
        Strengths("Science", 7),
        Strengths("Technology", 8),
        Strengths("Engineering", 7),
        Strengths("Business_and_Economics", 6),
        Strengths("History", 6),
        Strengths("Literature", 7),
        Strengths("Philosophy", 7),
        Strengths("Other", 6),
        Strengths("Vision", 9)  # Strong image processing and visual reasoning
    ],
    response_speed=6,  # Faster than large text models but slower than API models
    cost=0,  # Local model, no API cost
    vision_model=True
)

gpt4_vision = OnlineModel(
    name="gpt-4-turbo",
    vram=0,  # No VRAM needed for API calls
    strengths=[
        Strengths("NLP", 10),  # GPT-4 excels at Natural Language Processing
        Strengths("Coding", 8),  # GPT-4 is quite strong at coding tasks
        Strengths("Math", 9),  # GPT-4 is also very capable with math-related tasks
        Strengths("Science", 9),  # Strong in scientific understanding and research
        Strengths("Technology", 9),  # Strong in technology discussions
        Strengths("Engineering", 8),  # Solid knowledge in engineering fields
        Strengths("Business_and_Economics", 8),  # Good at analyzing business-related queries
        Strengths("History", 7),  # Good, but slightly less specialized in history
        Strengths("Literature", 8),  # Good for understanding literature and literary analysis
        Strengths("Philosophy", 8),  # Strong in philosophical discussions and critical thinking
        Strengths("Other", 7),  # Strong in various other topics
        Strengths("Vision", 10)  # GPT-4 Vision excels at image processing and visual reasoning
    ],
    response_speed=7,  # API models like GPT-4 Vision are generally fast but may still take longer due to the added complexity of vision tasks
    cost=300,  # Cost per 1,000,000 tokens for the 8K model (in USD) x 10
    vision_model=True  # Marking it as a vision model
)


_available_models = [deepseek_r1_671b, deepseek_r1_8b, deepseek_r1_1_5, tinyllama, gptturbo, o3mini, llama3_2_vision, gpt4_vision]

# TODO: Move to settings file

#

if __name__ == '__main__':

    record_audio()
    #
    # print(f"Prompt: {prompt}")
    #
    # #print(deepseek_r1_1_5.chat(prompt))
    #
    # print(gptturbo.chat(prompt))






