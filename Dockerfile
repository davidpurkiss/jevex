# jevex serve, the extraction microservice (README: "Microservice").
#
#   docker build -t jevex .
#   docker run -p 8080:8080 -e TYPESAFE_API_KEY -v jevex-data:/data jevex \
#       --schema carfinder.schemas:VehicleSpec --store sqlite:////data/jevex.db
#
# Schemas are imported by module path, so their package must be importable in the
# container: build an image FROM this one that pip installs it, or mount it and set
# PYTHONPATH. More extras: --build-arg EXTRAS=server,postgres,anthropic

FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
ARG EXTRAS=server
COPY pyproject.toml uv.lock README.md LICENSE ./
# Dependencies first, so changing the source doesn't reinstall them.
RUN extras=$(printf -- '--extra %s ' $(echo "$EXTRAS" | tr ',' ' ')) \
    && uv sync --locked --no-dev --no-install-project $extras
COPY src ./src
RUN extras=$(printf -- '--extra %s ' $(echo "$EXTRAS" | tr ',' ' ')) \
    && uv sync --locked --no-dev --no-editable $extras

FROM python:3.12-slim
RUN useradd --create-home --uid 1000 jevex && mkdir /data && chown jevex /data
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1
USER jevex
WORKDIR /home/jevex
VOLUME /data
EXPOSE 8080
# Assumes the default port; change it here too if you pass --port.
HEALTHCHECK --interval=30s --timeout=5s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4)"]
ENTRYPOINT ["jevex", "serve", "--host", "0.0.0.0", "--port", "8080"]
CMD ["--help"]
