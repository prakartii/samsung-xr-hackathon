# REWIND submission image: CPU only, fully offline at run time (ARCHITECTURE.md §13).
#
#   tools/fetch_weights.sh                 # fills weights/ first (incl. released_features)
#   docker build -t rewind .
#   docker run --rm -v <videos>:/in:ro -v <out>:/out rewind /in/P01_11.mp4 /out/P01_11.json
#
# The image holds code, config and weights only: no data, cache or annotations.
FROM python:3.11-slim

# ffmpeg: ffprobe for run.sh's probe (PyAV is the fallback); libgl1 + libglib2.0-0:
# OpenCV on a bare Debian image; libegl1 + libgles2: MediaPipe's Tasks library loads
# libEGL.so.1 at start even on CPU (CI caught its absence: hands failed to load).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY run.sh ./
COPY code ./code
COPY configs ./configs
COPY weights ./weights
RUN chmod +x run.sh

ENV PYTHONDONTWRITEBYTECODE=1 PYTHON=python3
ENTRYPOINT ["./run.sh"]
