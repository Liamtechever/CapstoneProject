from abc import ABC, abstractmethod
from typing import List

from ollama import ChatResponse, chat

from AudioRecording import record_audio
from SpeechToText import transcribe_audio


class Strengths:
    def __init__(self, subject_strength: str, strength_level: int):
        self.subject_strength = subject_strength
        self.strength_level = strength_level


class Model(ABC):
    # TODO: Speed
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
    def cost(self) -> int:
        pass

    @abstractmethod
    def chat(self, prompt: str, system_message: str = ""):
        pass


class LocalModel(Model):
    def __init__(self, name: str, vram: int, strengths: List[Strengths], cost: int):
        super().__init__(name)
        self._vram = vram
        self._strengths = strengths
        self._cost = cost

    @property
    def vram(self) -> int:
        return self._vram

    @property
    def strengths(self) -> List[Strengths]:
        return self._strengths

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
    def __init__(self, name: str, vram: int, strengths: List[Strengths], cost: int):
        super().__init__(name)
        self._vram = vram
        self._strengths = strengths
        self._cost = cost

    @property
    def vram(self) -> int:
        return self._vram

    @property
    def strengths(self) -> List[Strengths]:
        return self._strengths

    @property
    def cost(self) -> int:
        return self._cost

    def chat(self, prompt: str, system_message: str = ""):
        # Insert OpenAPI library calls here
        pass


# Example instantiation with sample data
deepseek_r1_8b = LocalModel(
    name="deepseek-r1:8b",
    vram=5,
    strengths=[Strengths("NLP", 9), Strengths("Reasoning", 8)],
    cost=50
)


deepseek_r1_1_5 = LocalModel(
    name="deepseek-r1:1.5b",
    vram=2,
    strengths=[Strengths("NLP", 9), Strengths("Reasoning", 8)],
    cost=50
)


tinyllama = LocalModel(
    name="tinyllama",
    vram=1,
    strengths=[],
    cost=0
)

if __name__ == '__main__':

    record_audio()
    prompt=transcribe_audio("output.mp3")

    print(f"Prompt: {prompt}")

    print(deepseek_r1_1_5.chat(prompt))





