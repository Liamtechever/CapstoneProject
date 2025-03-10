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
    vram=404,
    strengths=[Strengths("NLP", 9), Strengths("Coding", 9)],
    response_speed=8,
    cost=50
)

# Example instantiation with sample data
deepseek_r1_8b = LocalModel(
    name="deepseek-r1:8b",
    vram=5,
    strengths=[Strengths("NLP", 9), Strengths("Coding", 9)],
    response_speed=4,
    cost=50
)


deepseek_r1_1_5 = LocalModel(
    name="deepseek-r1:1.5b",
    vram=2,
    strengths=[Strengths("NLP", 9), Strengths("Reasoning", 8)],
    response_speed=2,
    cost=50
)


tinyllama = LocalModel(
    name="tinyllama",
    vram=1,
    strengths=[],
    response_speed=2,
    cost=0
)


gptturbo = OnlineModel(
    name="gpt-3.5-turbo",
    vram=1,
    strengths=[],
    response_speed=0,
    cost=0
)


_available_models = [deepseek_r1_671b, deepseek_r1_8b, deepseek_r1_1_5, tinyllama, gptturbo]

# TODO: Move to settings file

#

if __name__ == '__main__':

    record_audio()
    prompt=transcribe_audio("output.mp3")

    print(f"Prompt: {prompt}")

    #print(deepseek_r1_1_5.chat(prompt))

    print(gptturbo.chat(prompt))






