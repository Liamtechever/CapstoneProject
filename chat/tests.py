from unittest.mock import patch

import pytest
from django.test import Client
from django.urls import reverse

from core.errors import ProviderError

pytestmark = pytest.mark.django_db


@pytest.fixture
def signed_in(client, django_user_model):
    user = django_user_model.objects.create_user(
        username="test-user", password="test-only-password"
    )
    client.force_login(user)
    return client


def test_anonymous_chat_requires_login(client):
    response = client.get(reverse("chat"))
    assert response.status_code == 302
    assert "/login/" in response.url


def test_invalid_login_and_valid_login(client, django_user_model):
    django_user_model.objects.create_user(username="user", password="test-only-password")
    assert (
        client.post(reverse("login"), {"username": "user", "password": "wrong"}).status_code == 200
    )
    response = client.post(reverse("login"), {"username": "user", "password": "test-only-password"})
    assert response.status_code == 302
    assert client.get(reverse("chat")).status_code == 200


def test_submission_redirects_and_refresh_does_not_repeat_request(signed_in):
    with patch("chat.views.generate_response", return_value=("answer", "mock-model")) as generate:
        response = signed_in.post(reverse("chat"), {"message": "question"})
        assert response.status_code == 302
        signed_in.get(response.url)
        signed_in.get(response.url)
        generate.assert_called_once_with("question")
    assert len(signed_in.session["chat_history"]) == 2


@pytest.mark.parametrize("message", ["", "   ", "x" * 4001])
def test_invalid_message_never_calls_provider(signed_in, message):
    with patch("chat.views.generate_response") as generate:
        assert signed_in.post(reverse("chat"), {"message": message}).status_code == 400
        generate.assert_not_called()


def test_history_is_bounded(signed_in):
    with patch("chat.views.generate_response", return_value=("answer", "mock-model")):
        for index in range(12):
            signed_in.post(reverse("chat"), {"message": str(index)})
    assert len(signed_in.session["chat_history"]) == 20
    assert signed_in.session["chat_history"][0]["text"] == "2"


def test_provider_failure_keeps_history_consistent(signed_in):
    with patch(
        "chat.views.generate_response", side_effect=ProviderError("Local inference failed.")
    ):
        response = signed_in.post(reverse("chat"), {"message": "question"})
    assert response.status_code == 400
    assert "Local inference failed." in response.content.decode()
    assert signed_in.session.get("chat_history", []) == []


def test_unexpected_error_hides_sensitive_details(signed_in):
    with patch("chat.views.generate_response", side_effect=RuntimeError("private-provider-data")):
        response = signed_in.post(reverse("chat"), {"message": "question"})
    assert "private-provider-data" not in response.content.decode()


def test_messages_are_escaped(signed_in):
    with patch("chat.views.generate_response", return_value=("<script>bad()</script>", "model")):
        signed_in.post(reverse("chat"), {"message": "<img src=x onerror=bad()>"})
    html = signed_in.get(reverse("chat")).content.decode()
    assert "<script>bad()" not in html
    assert "<img src=x" not in html
    assert "&lt;script&gt;" in html


def test_sessions_are_isolated(signed_in, django_user_model):
    with patch("chat.views.generate_response", return_value=("private answer", "model")):
        signed_in.post(reverse("chat"), {"message": "private question"})
    other = Client()
    other.force_login(django_user_model.objects.create_user(username="other"))
    assert "private answer" not in other.get(reverse("chat")).content.decode()


def test_csrf_is_required_and_logout_is_post_only(django_user_model):
    client = Client(enforce_csrf_checks=True)
    client.force_login(django_user_model.objects.create_user(username="csrf-user"))
    assert client.post(reverse("chat"), {"message": "question"}).status_code == 403
    assert client.get(reverse("logout")).status_code == 405
    client.get(reverse("chat"))
    token = client.cookies["csrftoken"].value
    assert client.post(reverse("logout"), {"csrfmiddlewaretoken": token}).status_code == 302
    assert client.get(reverse("chat")).status_code == 302


def test_clear_history(signed_in):
    session = signed_in.session
    session["chat_history"] = [{"user": True, "text": "old"}]
    session.save()
    assert signed_in.post(reverse("chat"), {"action": "clear"}).status_code == 302
    assert "chat_history" not in signed_in.session


def test_unapproved_host_is_rejected(signed_in):
    assert signed_in.get(reverse("chat"), HTTP_HOST="attacker.example").status_code == 400
