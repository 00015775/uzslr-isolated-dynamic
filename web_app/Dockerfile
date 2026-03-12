# base image
# Python 3.9 — matches conda env exactly (python=3.9.23)
# slim variant keeps image lean; we add only what mediapipe + OpenCV need
FROM python:3.9-slim

# system dependencies 
# libglib2.0-0  -> mediapipe runtime
# libgl1        -> OpenCV (even headless build links against it)
# libgomp1      -> PyTorch CPU thread parallelism (OpenMP)
# libgcc-s1     -> mediapipe native libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libgcc-s1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# working directory
WORKDIR /app

# pip dependencies
# Copy requirements first so Docker layer-caches this step.
# Only re-runs when requirements.txt changes, not on every code change.
COPY web_app/requirements-docker.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# application source
# backend Python package
COPY web_app/backend/   /app/web_app/backend/
COPY web_app/__init__.py          /app/web_app/__init__.py

# make backend a proper Python package inside the image too
RUN touch /app/web_app/backend/__init__.py

# frontend static files
COPY web_app/frontend/  /app/web_app/frontend/

# model weights (6.7 MB)
COPY web_app/best_model.pth /app/web_app/best_model.pth

# pre-rendered sign videos
COPY web_app/sign_videos/       /app/web_app/sign_videos/

# sign GIFs served as static files
COPY show-50-signs/signs/ /app/show-50-signs/signs/

# runtime 
# Port 7860 is the default for Hugging Face Spaces (also fine for local Docker)
EXPOSE 7860

# Run from /app so all relative path resolution in main.py works correctly.
# web_app.backend.main resolves paths using __file__, so cwd doesn't matter,
# but starting here keeps things clean.
CMD ["uvicorn", "web_app.backend.main:app", "--host", "0.0.0.0", "--port", "7860"]