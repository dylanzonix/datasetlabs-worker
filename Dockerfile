FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for unstructured
RUN apt-get update && apt-get install -y \
    libmagic1 \
    libxml2 \
    libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY datasetlabs/ ./datasetlabs/
COPY worker/ ./worker/

# Run the worker
CMD ["python", "-m", "worker.main"]