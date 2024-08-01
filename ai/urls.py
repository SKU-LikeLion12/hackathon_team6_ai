from django.urls import path
from .views import transcribe_and_process

urlpatterns = [
    path('transcribe/', transcribe_and_process, name='transcribe_and_process'),
]