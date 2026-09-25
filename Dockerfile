FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /code

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen
RUN uv run python -m spacy download en_core_web_lg

COPY app ./app
COPY main.py ./

CMD ["uv", "run", "fastapi", "run", "main.py", "--port", "80"]