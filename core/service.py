"""One policy for classification, drafting, and revision; no import-time work."""

from dataclasses import replace

from core.classifier import Classification, classify_prompt
from core.config import load_settings
from core.errors import AssistantError
from core.models import _available_models
from core.picker import pick_model

MAX_PROMPT_LENGTH = 4000


def generate_response(message, settings=None, models=None, image=None):
    if not isinstance(message, str) or not 1 <= len(message.strip()) <= MAX_PROMPT_LENGTH:
        raise AssistantError(f"Enter a message between 1 and {MAX_PROMPT_LENGTH} characters.")
    message = message.strip()
    settings = settings or load_settings()
    models = _available_models if models is None else models
    classifier_model = pick_model(
        models, Classification("Other", 1, False, False), max_number_of_models=1, settings=settings
    )[0]
    classification = classify_prompt(message, model=classifier_model)
    if image:
        classification = replace(classification, requires_vision=True)
    if classification.requires_vision and not image:
        raise AssistantError(
            "This request needs an image. Use the command-line camera option; web chat is text only."
        )
    selected = pick_model(models, classification, settings=settings)
    draft = selected[0].chat(message, image=image)
    if len(selected) == 1:
        return draft, selected[0].name
    revised = selected[1].chat(
        f"Original request:\n{message}\n\nDraft answer:\n{draft}",
        system_message="Review the draft for correctness. Return only your revised answer. Keep it concise.",
        image=image,
    )
    return revised, selected[1].name
