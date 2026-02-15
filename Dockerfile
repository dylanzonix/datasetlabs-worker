FROM python:3.11-slim

WORKDIR /app

# System dependencies: document parsing + headful browser (Xvfb, fonts, Chrome libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Document parsing
    libmagic1 \
    libxml2 \
    libxslt1.1 \
    # Virtual display for headful Chrome
    xvfb \
    # D-Bus (Chrome expects it)
    dbus \
    # Fonts for real rendering (anti-detection)
    fonts-liberation \
    fonts-noto-color-emoji \
    fonts-noto-cjk \
    fontconfig \
    # Chrome headful dependencies
    libgbm1 \
    libxshmfence1 \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libgtk-3-0 \
    libdrm2 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    libxkbcommon0 \
    libx11-xcb1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright's Chromium (browser-use uses Playwright under the hood)
RUN python -m playwright install chromium \
    && python -m playwright install-deps chromium

# Copy API's dsl_api module (for shared models/db)
COPY api/dsl_api/ ./dsl_api/

# Copy worker code
COPY worker/dsl_worker/ ./dsl_worker/

# Copy entrypoint (starts Xvfb + dbus, then runs the worker)
COPY worker/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

CMD ["./entrypoint.sh"]
