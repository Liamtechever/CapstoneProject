from django import forms

from core.service import MAX_PROMPT_LENGTH


class ChatForm(forms.Form):
    message = forms.CharField(
        label="Your message",
        max_length=MAX_PROMPT_LENGTH,
        strip=True,
        widget=forms.Textarea(
            attrs={
                "rows": 3,
                "placeholder": "Ask a text question…",
                "aria-describedby": "message-help",
            }
        ),
    )
