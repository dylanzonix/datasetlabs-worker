FROM python:3.11-slim

WORKDIR /app

# System dependencies: document parsing only
# Browser runs on Browser Use Cloud — no local Chrome/Xvfb needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Document parsing
    libmagic1 \
    libxml2 \
    libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# Copy sandbox-service package (local dependency — client library for sandbox API)
COPY sandbox/pyproject.toml sandbox/README.md ./sandbox-service/
COPY sandbox/sandbox_service/ ./sandbox-service/sandbox_service/
RUN pip install --no-cache-dir ./sandbox-service

# Copy requirements and install Python dependencies
COPY worker/requirements.txt .
RUN grep -v '^sandbox-service' requirements.txt > requirements-filtered.txt && \
    pip install --no-cache-dir -r requirements-filtered.txt

# Copy API's dsl_api module (for shared models/db)
COPY api/dsl_api/ ./dsl_api/

# Copy worker code
COPY worker/dsl_worker/ ./dsl_worker/

# Copy entrypoint
COPY worker/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

CMD ["./entrypoint.sh"]
