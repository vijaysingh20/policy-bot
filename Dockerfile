# Backend image for Hugging Face Spaces (Docker SDK, free CPU tier).
FROM python:3.12-slim

# uv binary from the official image — pin this to the version you use locally
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Spaces run the container as UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/app/.venv/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# 1) Dependencies only — this layer is cached until pyproject.toml / uv.lock change
COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 2) Bake both models into the image so a cold start doesn't need to download them
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')"

# 3) Application code last — editing code doesn't reinstall dependencies
COPY --chown=user backend ./backend
RUN mkdir -p storage db uploads logs

EXPOSE 7860
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "7860"]