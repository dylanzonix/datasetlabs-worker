FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for unstructured (only if you need file parsing)
RUN apt-get update && apt-get install -y \
    libmagic1 \
    libxml2 \
    libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API's dsl_api module (for shared models/db)
COPY api/dsl_api/ ./dsl_api/

# Copy worker code
COPY worker/dsl_worker/ ./dsl_worker/

# Run the worker
CMD ["python", "-m", "dsl_worker.main"]