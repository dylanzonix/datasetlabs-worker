#!/bin/bash
set -e

# Run the worker
exec python -m dsl_worker.main
