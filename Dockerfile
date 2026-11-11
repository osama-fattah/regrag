FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[openai,bedrock]"

COPY data ./data
# Build the offline index at image build time; set REGRAG_EMBEDDINGS at runtime and
# re-run `regrag ingest` to switch to Bedrock / Azure / OpenAI embeddings.
RUN regrag ingest --src data/sample_corpus

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "regrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
