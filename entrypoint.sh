#!/bin/bash
set -e

# Disable browser-use telemetry
export ANONYMIZED_TELEMETRY=false

# Start virtual display (headful Chrome requires an X server)
Xvfb :99 -screen 0 1920x1080x24 -nolisten tcp &
export DISPLAY=:99

# Wait for Xvfb to be ready
sleep 1

# Start dbus (Chrome expects it)
mkdir -p /run/dbus
dbus-daemon --system --fork 2>/dev/null || true

# Run the worker
exec python -m dsl_worker.main
