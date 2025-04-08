from django.urls import path
from .views import ChatView
from django.contrib.auth.decorators import login_required

urlpatterns = [
    path('', login_required(ChatView.as_view()), name='chat'),
]
