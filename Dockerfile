FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LOAN_ML_ROOT=/app \
    LOAN_ML_STATE_DIR=/app/var

WORKDIR /app
RUN useradd --create-home --uid 10001 app

COPY requirements.txt pyproject.toml README.md ./
RUN pip install -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY reports ./reports
COPY models ./models
COPY app ./app
COPY .streamlit ./.streamlit
RUN pip install --no-deps -e . && mkdir -p /app/var && chown -R app:app /app/var

USER app
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD \
  python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

# Default: the API. docker-compose runs the dashboard from the same image.
CMD ["uvicorn", "loan_approval.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]
