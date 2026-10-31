FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN uv sync --no-dev --all-extras

# Indexes are built outside the image and mounted: docker run -v $PWD/data:/app/data ...
COPY app ./app
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "filings_copilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
