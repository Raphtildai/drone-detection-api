# Cloud Run image for the Drone Detection v2 API.
#
# Built to run the exact same app_v2.py used under cPanel/Passenger, but on
# infrastructure with real per-container CPU/memory limits instead of a
# shared account-wide process cap. async_mode stays "threading" (unchanged
# from app_v2.py) so this mirrors the existing polling-only Socket.IO
# behavior rather than risking eventlet + PyTorch interaction issues.
FROM python:3.12-slim

# libsndfile1 is required by soundfile/librosa at import time (not a pip
# package — it's the native libsndfile the Python bindings wrap).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Thread-pool env vars (mirrors the app_v2.py OpenBLAS/OMP fix) — harmless
# here since each Cloud Run container gets its own dedicated CPU quota, but
# keeps behavior identical to the cPanel deployment.
ENV OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

COPY requirements-prod.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-prod.txt \
    && pip install --no-cache-dir gunicorn==23.0.0

COPY app_v2.py .
COPY realtime_sessions.py .
COPY real_time_audio_v2.py .
COPY drone_detection/ ./drone_detection/
COPY models/ ./models/
COPY templates/ ./templates/

# Cloud Run injects $PORT (defaults to 8080) and expects the container to
# listen on it. --threads gives Flask-SocketIO's threading async_mode room
# for concurrent long-polling connections within one container; Cloud Run
# itself scales the number of containers under load.
CMD exec gunicorn --bind :${PORT:-8080} --workers 1 --threads 8 --timeout 300 app_v2:app
