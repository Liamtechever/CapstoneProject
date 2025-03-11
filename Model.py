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

openai.api_key = os.getenv("OPEN_API_KEY")

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
    def chat(self, prompt: str, system_message: str = ""):
        pass

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return self.__str__()

num = 8

text = str(num)


class LocalModel(Model):
    def __init__(self, name: str, vram: int, strengths: List[Strengths], response_speed: int, cost: int):
        super().__init__(name)
        self._vram = vram
        self._strengths = strengths
        self._speed = response_speed
        self._cost = cost

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

    def chat(self, prompt: str, system_message: str = ""):
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
    def __init__(self, name: str, vram: int, strengths: List[Strengths], response_speed: int, cost: int):
        super().__init__(name)
        self._vram = vram
        self._strengths = strengths
        self._response_speed = speed
        self._cost = cost

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

    def chat(self, prompt: str, system_message: str = ""):
        # Insert OpenAPI library calls here


        response = openai.chat.completions.create(
            model=self.name,  # Or "gpt-3.5-turbo" if GPT-4 is unavailable
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"{prompt}"},
            ],
            max_tokens=50,
            temperature=0.7,
        )

        response_text = response.choices[0].message.content.strip()


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
    cost=0  # Local model, no API cost
)

_available_models = [deepseek_r1_671b, deepseek_r1_8b, deepseek_r1_1_5, tinyllama, gptturbo, o3mini]

# TODO: Move to settings file

#

if __name__ == '__main__':

    record_audio()
    prompt=transcribe_audio("output.mp3")

    print(f"Prompt: {prompt}")

    #print(deepseek_r1_1_5.chat(prompt))

    print(gptturbo.chat(prompt))






