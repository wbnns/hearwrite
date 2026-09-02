# HearWrite on a CPU only box. No GPU, no CUDA, no torch.
#
# Models are downloaded and pruned AT BUILD TIME, so a container starts in about
# a second rather than fetching 600MB on its first request. That trades image
# size for predictable startup, which is the right way round for a service. To
# trade it back, drop the download layer and mount a volume at
# /root/.cache/hearwrite instead.
FROM python:3.13-slim

# ffmpeg is only needed to build evaluation fixtures, not to run the service.
# Nothing else is required: every model is ONNX and runs on the CPU.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir '.[onnx,turn,server]'

# Fetch the weights, then delete the float builds we never load. That is about
# 350MB of a 600MB download.
#
# `hearwrite models` only LISTS; ensure() is what downloads. Each file is
# checked against its pinned SHA-256 and the build fails if one does not match,
# which is the point of pinning them.
RUN python -c "from hearwrite.models import REGISTRY, ensure; \
[ensure(REGISTRY[n]) for n in ('zipformer-en','titanet-small','silero-vad','smart-turn')]" \
    && hearwrite models --prune

EXPOSE 8080

# 0.0.0.0 because a container's loopback is not reachable from outside it.
# Put a TLS terminator in front: HearWrite speaks plain WebSocket and does not
# manage certificates.
CMD ["hearwrite", "serve", "--host", "0.0.0.0", "--port", "8080", "--policy", "conversation"]
