"""Authenticated text-only chat with bounded display history."""

import logging

from django.shortcuts import redirect, render
from django.views import View

from chat.forms import ChatForm
from core.errors import AssistantError
from core.service import generate_response

logger = logging.getLogger(__name__)
MAX_HISTORY_MESSAGES = 20


class ChatView(View):
    def get(self, request):
        return render(
            request,
            "chat/chat.html",
            {
                "form": ChatForm(),
                "history": request.session.get("chat_history", []),
            },
        )

    def post(self, request):
        if request.POST.get("action") == "clear":
            request.session.pop("chat_history", None)
            return redirect("chat")
        form = ChatForm(request.POST)
        if form.is_valid():
            try:
                output, model = generate_response(form.cleaned_data["message"])
            except AssistantError as exc:
                form.add_error(None, str(exc))
            except Exception:
                # Omit prompts, provider response bodies, and exception text.
                logger.error("Unexpected chat generation failure")
                form.add_error(None, "The response could not be generated. Please try again.")
            else:
                history = request.session.get("chat_history", [])
                history.extend(
                    [
                        {"user": True, "text": form.cleaned_data["message"]},
                        {"user": False, "text": output, "model": model},
                    ]
                )
                request.session["chat_history"] = history[-MAX_HISTORY_MESSAGES:]
                return redirect("chat")
        return render(
            request,
            "chat/chat.html",
            {
                "form": form,
                "history": request.session.get("chat_history", []),
            },
            status=400,
        )
