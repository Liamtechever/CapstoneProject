"""Lazy provider adapters. Registry scores are inherited heuristics, not benchmarks."""

import base64
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from core.config import load_environment
from core.errors import ConfigurationError, ProviderError


@dataclass(frozen=True)
class Strengths:
    subject_strength: str
    strength_level: int


class Model(ABC):
    def __init__(self, name, vram, strengths, response_speed, cost, vision_model=False):
        self.name = name
        self.vram = vram
        self.strengths = strengths
        self.response_speed = response_speed
        self.cost = cost
        self.vision_model = vision_model

    def __repr__(self):
        return self.name

    @abstractmethod
    def chat(self, prompt, system_message="", image=None):
        raise NotImplementedError


def _image_data(path):
    path = Path(path)
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ConfigurationError("Use a JPEG or PNG image.")
    try:
        if path.stat().st_size > 8 * 1024 * 1024:
            raise ConfigurationError("Images must be at most 8 MB.")
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigurationError("The image could not be read.") from exc
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


class LocalModel(Model):
    def chat(self, prompt, system_message="", image=None):
        if image and not self.vision_model:
            raise ConfigurationError("The selected model does not accept images.")
        from ollama import Client

        user = {"role": "user", "content": prompt}
        if image:
            _image_data(image)  # Validate size and readability before the request.
            user["images"] = [str(image)]
        try:
            client = Client(host="http://127.0.0.1:11434", timeout=60)
            response = client.chat(
                model=self.name,
                messages=[{"role": "system", "content": system_message}, user],
                options={"num_predict": 512},
            )
            text = response.message.content
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Empty response")
            return text.strip()[:12000]
        except Exception as exc:
            raise ProviderError(
                "Local inference failed. Check that Ollama is running and the selected model is installed."
            ) from exc


class OnlineModel(Model):
    def chat(self, prompt, system_message="", image=None):
        if image and not self.vision_model:
            raise ConfigurationError("The selected model does not accept images.")
        load_environment()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "Hosted models require OPENAI_API_KEY in your environment or .env."
            )
        from openai import OpenAI

        content = prompt
        if image:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _image_data(image)}},
            ]
        try:
            with OpenAI(api_key=api_key, timeout=60, max_retries=0) as client:
                response = client.chat.completions.create(
                    model=self.name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": content},
                    ],
                    max_completion_tokens=1024,
                )
            text = response.choices[0].message.content
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Empty response")
            return text.strip()[:12000]
        except Exception as exc:
            raise ProviderError(
                "Hosted inference failed. Check account access, quota, and model availability."
            ) from exc


# Original registry and scores retained for attribution. These are not measured prices
# or benchmarks. Unknown memory requirements exclude the 671B entry from routing.
deepseek_r1_671b = LocalModel(
    name="deepseek-r1:671b",
    vram=None,
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
        Strengths("Other", 7),
    ],
    response_speed=10,
    cost=0,
)
deepseek_r1_8b = LocalModel(
    name="deepseek-r1:8b",
    vram=16,
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
        Strengths("Other", 6),
    ],
    response_speed=7,
    cost=0,
)
deepseek_r1_1_5 = LocalModel(
    name="deepseek-r1:1.5b",
    vram=4,
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
        Strengths("Other", 6),
    ],
    response_speed=5,
    cost=0,
)
tinyllama = LocalModel(
    name="tinyllama",
    vram=2,
    strengths=[
        Strengths("Coding", 9),
        Strengths("Technology", 7),
        Strengths("Engineering", 6),
        Strengths("Other", 5),
    ],
    response_speed=3,
    cost=0,
)
gptturbo = OnlineModel(
    name="gpt-3.5-turbo",
    vram=0,
    strengths=[
        Strengths("Coding", 5),
        Strengths("Technology", 5),
        Strengths("Business_and_Economics", 5),
        Strengths("History", 5),
        Strengths("Literature", 5),
        Strengths("Philosophy", 5),
        Strengths("Other", 6),
    ],
    response_speed=0,
    cost=5,
)
o3mini = OnlineModel(
    name="o3-mini",
    vram=0,
    strengths=[
        Strengths("Coding", 5),
        Strengths("Technology", 5),
        Strengths("Business_and_Economics", 5),
        Strengths("History", 5),
        Strengths("Literature", 5),
        Strengths("Philosophy", 5),
        Strengths("Other", 6),
    ],
    response_speed=0,
    cost=3,
)
llama3_2_vision = LocalModel(
    name="llama3.2-vision",
    vram=24,
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
        Strengths("Vision", 9),
    ],
    response_speed=6,
    cost=0,
    vision_model=True,
)
gpt4_vision = OnlineModel(
    name="gpt-4-turbo",
    vram=0,
    strengths=[
        Strengths("NLP", 10),
        Strengths("Coding", 8),
        Strengths("Math", 9),
        Strengths("Science", 9),
        Strengths("Technology", 9),
        Strengths("Engineering", 8),
        Strengths("Business_and_Economics", 8),
        Strengths("History", 7),
        Strengths("Literature", 8),
        Strengths("Philosophy", 8),
        Strengths("Other", 7),
        Strengths("Vision", 10),
    ],
    response_speed=7,
    cost=10,  # Relative preference score; normalized to the settings scale, not dollars.
    vision_model=True,
)
_available_models = [
    deepseek_r1_671b,
    deepseek_r1_8b,
    deepseek_r1_1_5,
    tinyllama,
    gptturbo,
    o3mini,
    llama3_2_vision,
    gpt4_vision,
]
