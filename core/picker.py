"""Apply the same constraints to text, classification, and image requests."""

from core.config import load_settings
from core.errors import NoEligibleModel
from core.models import OnlineModel


def pick_model(available_models, prompt_classification, max_number_of_models=2, settings=None):
    settings = settings or load_settings()
    if type(max_number_of_models) is not int or max_number_of_models < 1:
        raise ValueError("max_number_of_models must be positive")
    eligible = [
        model
        for model in available_models
        if model.vram is not None
        and model.vram <= settings.user_vram
        and model.cost <= settings.api_cost_tolerance
        and (settings.use_internet or not isinstance(model, OnlineModel))
        and (not prompt_classification.requires_vision or model.vision_model)
    ]
    if not eligible:
        raise NoEligibleModel(
            "No model meets the configured memory, internet, cost, and vision limits."
        )

    def strength(model):
        return next(
            (
                s.strength_level
                for s in model.strengths
                if s.subject_strength == prompt_classification.subject
            ),
            0,
        )

    return sorted(eligible, key=lambda model: (-strength(model), model.cost, model.name))[
        :max_number_of_models
    ]
