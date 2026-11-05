from unittest.mock import Mock

import pytest

from core.classifier import Classification, parse_classification
from core.config import Settings
from core.errors import AssistantError, NoEligibleModel
from core.models import LocalModel, OnlineModel, Strengths
from core.picker import pick_model
from core.service import generate_response


def local(name="local", memory=2, vision=False, strength=5):
    model = LocalModel(name, memory, [Strengths("Other", strength)], 3, 0, vision)
    model.chat = Mock(return_value="draft")
    return model


def online():
    model = OnlineModel("hosted", 0, [Strengths("Other", 10)], 2, 5)
    model.chat = Mock(return_value="online")
    return model


def classification(vision=False):
    return Classification("Other", 1, False, vision)


def test_vision_obeys_memory_and_network_policy():
    with pytest.raises(NoEligibleModel):
        pick_model([local(memory=24, vision=True)], classification(True), settings=Settings())
    hosted = online()
    hosted.vision_model = True
    with pytest.raises(NoEligibleModel):
        pick_model([hosted], classification(True), settings=Settings(api_cost_tolerance=10))


def test_empty_candidates_produce_explicit_error():
    with pytest.raises(NoEligibleModel):
        pick_model([], classification())


def test_unknown_memory_excluded():
    with pytest.raises(NoEligibleModel):
        pick_model([local(memory=None)], classification())


@pytest.mark.parametrize("budget,permitted", [(4, False), (5, True)])
def test_online_cost_boundary(budget, permitted):
    settings = Settings(use_internet=True, api_cost_tolerance=budget)
    if permitted:
        assert pick_model([online()], classification(), settings=settings)
    else:
        with pytest.raises(NoEligibleModel):
            pick_model([online()], classification(), settings=settings)


def test_speed_properties_are_constructor_values():
    assert local().response_speed == 3
    assert online().response_speed == 2


def test_offline_pipeline_never_calls_online_provider():
    first, second, hosted = local("first", strength=7), local("second"), online()
    first.chat.side_effect = [
        '{"subject":"Other","difficulty":1,"requires_thinking":false,"requires_vision":false}',
        "draft",
    ]
    second.chat.return_value = "reviewed"
    assert generate_response("question", Settings(), [first, second, hosted]) == (
        "reviewed",
        "second",
    )
    hosted.chat.assert_not_called()
    assert first.chat.call_count == 2
    assert second.chat.call_count == 1
    assert "draft" in second.chat.call_args.args[0]


def test_one_model_has_no_duplicate_review_call():
    model = local()
    model.chat.side_effect = [
        '{"subject":"Other","difficulty":1,"requires_thinking":false,"requires_vision":false}',
        "answer",
    ]
    assert generate_response("question", Settings(), [model]) == ("answer", "local")
    assert model.chat.call_count == 2


def test_web_vision_request_does_not_invent_an_image():
    model = local()
    model.chat.return_value = (
        '{"subject":"Other","difficulty":1,"requires_thinking":false,"requires_vision":true}'
    )
    with pytest.raises(AssistantError, match="needs an image"):
        generate_response("look", Settings(), [model])
    assert model.chat.call_count == 1


@pytest.mark.parametrize(
    "response",
    [
        '{"subject":"Other","difficulty":99,"requires_thinking":false,"requires_vision":false}',
        '{"subject":"Other","difficulty":1,"requires_thinking":false,"requires_vision":"false"}',
        '{"subject":"Other","difficulty":true,"requires_thinking":false,"requires_vision":false}',
        '{"subject":"Unknown","difficulty":1,"requires_thinking":false,"requires_vision":false}',
        "{}",
        "[]",
        "not JSON",
        '{"subject":"Other","difficulty":1,"requires_thinking":false,"requires_vision":false,"extra":1}',
    ],
)
def test_invalid_classifier_outputs_rejected(response):
    with pytest.raises(AssistantError):
        parse_classification(response)


def test_valid_fenced_classifier_output():
    result = parse_classification(
        '```json\n{"subject":"Math","difficulty":2,"requires_thinking":true,"requires_vision":false}\n```'
    )
    assert result == Classification("Math", 2, True, False)


@pytest.mark.parametrize("prompt", ["", "  ", "x" * 4001, None])
def test_invalid_prompt_causes_no_provider_call(prompt):
    model = local()
    with pytest.raises(AssistantError):
        generate_response(prompt, models=[model])
    model.chat.assert_not_called()
