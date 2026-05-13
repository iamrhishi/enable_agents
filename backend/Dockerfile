# Backend Dockerfile - Python Flask Application
# Pin Debian 12 (bookworm): unqualified python:3.11-slim tracks newer Debian (trixie) which
# can make `apt-get` flaky on some hosts/mirrors during remote builds.
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update -o Acquire::Retries=5 \
    && apt-get install -y --no-install-recommends \
        gcc \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake NLTK data into the image (avoids flaky runtime downloads and worker races on punkt_tab/stopwords).
RUN python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Copy application code
COPY . .

# Create instance directory for data
RUN mkdir -p /app/instance/logs

# Expose port (overridden per profile in docker-compose)
EXPOSE 8000

# No HEALTHCHECK here — defined per-service in docker-compose so that
# celery/beat containers (which have no HTTP server) are not checked.

# Run the application (overridden by docker-compose command per profile)
CMD ["sh", "-c", "flask db upgrade 2>/dev/null || true && python app.py"]
