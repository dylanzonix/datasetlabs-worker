FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for unstructured (only if you need file parsing)
RUN apt-get update && apt-get install -y \
    libmagic1 \
    libxml2 \
    libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY dsl_worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire API repo (for shared models)
COPY api/ ./api/

# Copy worker code
COPY dsl_worker/ ./worker/

# Run the worker
CMD ["python", "-m", "worker.main"]