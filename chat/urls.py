from django.contrib.auth.decorators import login_required
from django.urls import path

from .views import ChatView

urlpatterns = [
    path("", login_required(ChatView.as_view()), name="chat"),
]
