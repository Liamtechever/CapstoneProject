import json
import re
from dataclasses import dataclass

from AudioRecording import record_audio
from Model import deepseek_r1_1_5, gptturbo
from SpeechToText import transcribe_audio


@dataclass
class Classification:
    subject: str
    difficulty: int
    requires_thinking: bool
    requires_vision: bool





def classify_prompt(prompt: str):
    system_message = (
             "You are an AI assistant that classifies prompts into three categories with high accuracy:\n\n"
        "1. **Subject**: Identify the primary academic category of the prompt. Choose from a fixed list:\n"
        "   - 'Math'\n"
        "   - 'Science'\n"
        "   - 'History'\n"
        "   - 'Literature'\n"
        "   - 'Philosophy'\n"
        "   - 'Technology'\n"
        "   - 'Engineering'\n"
        "   - 'Business_and_Economics'\n"
        "   - 'NLP'\n"
        "   - 'Other' (if the prompt does not fit into a clear category)\n\n"
        "2. **Difficulty**: Assign a difficulty level based on complexity:\n"
        "   - 1 (Easy): Basic facts, definitions, simple arithmetic, or yes/no questions.\n"
        "   - 2 (Moderate): Requires some explanation, multi-step reasoning, or understanding of a concept.\n"
        "   - 3 (Difficult): Involves deep reasoning, derivations, complex problem-solving, or critical analysis.\n\n"
        "3. **Requires Thinking**: Determine whether the prompt requires reasoning or problem-solving.\n"
        "   - True: If answering requires logical reasoning, synthesis of information, or analytical thinking.\n"
        "   - False: If the answer is straightforward, fact-based, or requires minimal thought.\n\n"
         "4. **Requires Vision**: Determine whether the prompt requires analyzing an image, diagram, or visual content.\n"
        "   - True: If answering requires interpreting visual data, charts, diagrams, images or explicitly STATED by USER that Vision is wanted.\n"
        "   - False: If the answer is purely text-based and does not rely on visual input.\n\n"

        "### **Formatting Instructions:**\n"
        "Output the classification as JSON with keys: `subject`, `difficulty`,`requires_thinking`, and `requires_vision`.\n"
        "Make SURE to include all keys. ONLY use the subjects listed above."
        "YOU MUST STICK TO THE FORMAT OF THE EXAMPLE OUTPUT"
        "Example output:\n"
        '{ "subject": "Science", "difficulty": 2, "requires_thinking": true, "requires_vision": false }\n\n"'
        )

    response = deepseek_r1_1_5.chat(prompt=f"Classify this prompt: {prompt}", system_message=system_message)
    #response = gptturbo.chat(prompt=f"Classify this prompt: {prompt}", system_message=system_message)

    match = re.search(r"{[\s\S]*?}", response)

    if match: json_string = match.group(0)  # Extract the JSON portion print(json_string) else: print("No JSON found.")
    else: json_string = "{}"
    # Convert to dictionary
    try:
        c_dict = json.loads(json_string)
        classification = Classification(c_dict["subject"], c_dict["difficulty"], c_dict["requires_thinking"], c_dict["requires_vision"])
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
