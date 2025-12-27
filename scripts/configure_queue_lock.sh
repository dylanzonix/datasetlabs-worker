#!/usr/bin/env bash
# configure_queue_lock.sh - Configure Service Bus queue for long-running jobs
#
# This script increases the lock duration on the Service Bus queue to accommodate
# longer processing times. Combined with auto-lock renewal in the worker code,
# this allows jobs to run for extended periods without timing out.

set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-datasetlabs-rg}"
SERVICE_BUS_NAMESPACE="${SERVICE_BUS_NAMESPACE:-datasetlabs-bus}"
QUEUE_NAME="${QUEUE_NAME:-jobs}"

# Lock duration in ISO 8601 format
# PT5M = 5 minutes (maximum allowed at queue level)
# The worker's auto-renewal will extend this further as needed
LOCK_DURATION="${LOCK_DURATION:-PT5M}"

echo "Configuring Service Bus queue for long-running jobs..."
echo "Resource Group: $RESOURCE_GROUP"
echo "Namespace: $SERVICE_BUS_NAMESPACE"
echo "Queue: $QUEUE_NAME"
echo "Lock Duration: $LOCK_DURATION"
echo ""

# Update queue with longer lock duration
az servicebus queue update \
  --resource-group "$RESOURCE_GROUP" \
  --namespace-name "$SERVICE_BUS_NAMESPACE" \
  --name "$QUEUE_NAME" \
  --lock-duration "$LOCK_DURATION"

echo ""
echo "✅ Queue configured successfully!"
echo ""
echo "Current queue settings:"
az servicebus queue show \
  --resource-group "$RESOURCE_GROUP" \
  --namespace-name "$SERVICE_BUS_NAMESPACE" \
  --name "$QUEUE_NAME" \
  --query "{lockDuration: lockDuration, maxDeliveryCount: maxDeliveryCount, defaultMessageTimeToLive: defaultMessageTimeToLive}" \
  -o json

echo ""
echo "Note: Worker code has max_lock_renewal_duration=3600 (1 hour)"
echo "This means the worker will automatically renew the lock every few minutes"
echo "for up to 1 hour of processing time per message."