import json
import re
from dataclasses import dataclass

from AudioRecording import record_audio
from Model import deepseek_r1_1_5
from SpeechToText import transcribe_audio


@dataclass
class Classification:
    subject: str
    difficulty: int
    requires_thinking: bool





def classify_prompt(prompt: str):
    system_message = (
            "You are an assistant that classifies prompts into three categories:\n"
        "- Subject: The main category of the prompt (e.g., 'Science', 'Math', 'History').\n"
            # TODO: Clarify strictness of subjects and supported subjects
        "- Difficulty: A numerical value representing difficulty (e.g., 1 for easy, 2 for moderate, 3 for difficult).\n"
        "- Requires Thinking: True if the prompt involves reasoning or problem-solving, False otherwise.\n\n"
        "Format your output as JSON like this:\n"
        '{"subject": "Science", "difficulty": 2, "requires_thinking": true}\n\n'
        "Examples:\n"
        "- 'Explain Newton’s laws of motion.' -> {'subject': 'Science', 'difficulty': 2, 'requires_thinking': True}\n"
        "- 'What is 2 + 2?' -> {'subject': 'Math', 'difficulty': 1, 'requires_thinking': False}\n"
        "- 'Derive the formula for the area of a circle.' -> {'subject': 'Math', 'difficulty': 3, 'requires_thinking': True}"
        )

    response = deepseek_r1_1_5.chat(prompt=f"Classify this prompt: {prompt}", system_message=system_message)

    clean_response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL).strip()

    # Convert to dictionary
    try:
        c_dict = json.loads(clean_response)
        classification = Classification(c_dict["subject"], c_dict["difficulty"], c_dict["requires_thinking"])
    except json.JSONDecodeError:
        classification = Classification("None", 0, False)
        print("Error: Could not parse JSON.")

    print(response)

    # TODO: Implement GOOD error handling. This is AI we're talking about
    return classification

if __name__ == '__main__':
    record_audio()
    _prompt = transcribe_audio("output.mp3")

    print(f"Prompt: {_prompt}")

    print(classify_prompt(prompt=_prompt))
