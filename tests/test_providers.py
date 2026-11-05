from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.errors import ConfigurationError, ProviderError
from core.models import LocalModel, OnlineModel


def test_image_request_has_typed_content_blocks(tmp_path, monkeypatch):
    import openai

    import core.models as models

    monkeypatch.setattr(models, "load_environment", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-placeholder")
    client = MagicMock()
    client.__enter__.return_value = client
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="image answer"))]
    )
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(openai, "OpenAI", factory)
    image = tmp_path / "image.jpg"
    image.write_bytes(b"test image bytes")
    model = OnlineModel("vision", 0, [], 1, 1, True)
    assert model.chat("look", image=image) == "image answer"
    args = client.chat.completions.create.call_args.kwargs
    assert args["messages"][1]["content"][1]["type"] == "image_url"
    assert args["messages"][1]["content"][1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )
    assert factory.call_args.kwargs["max_retries"] == 0


def test_missing_key_is_checked_only_on_hosted_call(monkeypatch):
    import core.models as models

    monkeypatch.setattr(models, "load_environment", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    model = OnlineModel("hosted", 0, [], 1, 1)
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        model.chat("question")


def test_local_error_does_not_expose_provider_message(monkeypatch):
    import ollama

    client = MagicMock()
    client.chat.side_effect = RuntimeError("private prompt or credential")
    monkeypatch.setattr(ollama, "Client", MagicMock(return_value=client))
    with pytest.raises(ProviderError) as caught:
        LocalModel("local", 2, [], 1, 0).chat("question")
    assert "private" not in str(caught.value)


def test_local_vision_preserves_system_instruction(tmp_path, monkeypatch):
    import ollama

    client = MagicMock()
    client.chat.return_value = SimpleNamespace(message=SimpleNamespace(content="answer"))
    monkeypatch.setattr(ollama, "Client", MagicMock(return_value=client))
    image = tmp_path / "test.png"
    image.write_bytes(b"image")
    LocalModel("vision", 2, [], 1, 0, True).chat("look", system_message="instruction", image=image)
    assert client.chat.call_args.kwargs["messages"][0]["content"] == "instruction"
    assert client.chat.call_args.kwargs["messages"][1]["images"] == [str(image)]
