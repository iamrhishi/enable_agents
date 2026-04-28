# Backend Dockerfile - Python Flask Application
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
