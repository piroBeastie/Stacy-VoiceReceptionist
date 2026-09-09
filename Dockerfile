# Worker image. The worker dials *out* to LiveKit and registers -- it is not a
# server, listens on no port, and needs no inbound rule anywhere. That makes it
# deployable on anything that can run a container with egress.
#
# Two things here exist purely to keep the first second of a call quiet:
#
#   1. Model weights are downloaded at build time, not first-call time. Silero
#      and the end-of-utterance transformer are a disk read and a session build;
#      paid during `docker build` they cost nothing, paid on a cold worker they
#      are paid while somebody is listening to silence.
#   2. Nothing installs at runtime. A container that pip-installs on boot turns
#      a deploy into a network gamble.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ffmpeg/libgomp are what the audio and ONNX runtime paths reach for. Installing
# them before the source copy keeps this layer cached across code changes.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Dependencies before source, same reason: editing an agent file should not
# re-resolve the dependency tree.
COPY pyproject.toml README.md ./
COPY src/dana/__init__.py src/dana/__init__.py
RUN pip install --no-cache-dir -e .

COPY src/ src/
COPY scripts/ scripts/

# Download the VAD and turn-detector weights into the image. `download-files` is
# the SDK's own hook for exactly this; if it moves, the build fails loudly here
# rather than the first caller waiting through a model fetch.
RUN python -m dana.agent.entrypoint download-files

# A non-root user, because this process handles call audio and writes
# transcripts. /app/data is where both land.
RUN useradd --create-home --uid 10001 dana \
    && mkdir -p /app/data \
    && chown -R dana:dana /app
USER dana

ENV DANA_TENANT_DIR=/app/src/dana/tenants/data \
    DANA_DATA_DIR=/app/data \
    DANA_TRACE_DIR=/app/data/traces

# Bookings, transcripts and traces are on disk. Mount this or lose them on every
# redeploy -- and note that the calendar's lock is per-process, so a shared
# volume across two workers does not make them safe to run together. One worker
# per tenant calendar until that store moves behind Postgres.
VOLUME ["/app/data"]

# Fails the container if credentials or tenants are wrong, before a call arrives.
HEALTHCHECK --interval=60s --timeout=20s --start-period=30s --retries=2 \
    CMD python scripts/preflight.py > /dev/null || exit 1

# Greetings are deliberately NOT pre-rendered here. It would need vendor API
# keys at build time, and a key baked into an image leaks to anyone who can pull
# it. Run scripts/prerender_greetings.py as a deploy step instead; the worker
# renders on first use regardless, so a skipped step costs one caller ~400ms,
# not correctness.

# `start` is the production mode: register with LiveKit and wait for jobs.
# Override with `dev` for autoreload, or `console` to talk to it over your mic.
CMD ["python", "-m", "dana.agent.entrypoint", "start"]
