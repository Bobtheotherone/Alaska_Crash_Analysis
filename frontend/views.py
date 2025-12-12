from django.shortcuts import render


def index(request):
    # Single-page application entrypoint
    return render(request, "frontend/index.html")
