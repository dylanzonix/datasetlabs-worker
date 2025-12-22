#!/bin/bash
# keda-config.sh - Update KEDA polling and cooldown settings

set -e

RESOURCE_GROUP="datasetlabs-rg"
CONTAINER_APP_NAME="datasetlabs-worker"
POLLING_INTERVAL=10
COOLDOWN=300

echo "Setting KEDA polling interval to ${POLLING_INTERVAL}s and cooldown to ${COOLDOWN}s..."

SUBSCRIPTION_ID=$(az account show --query id -o tsv)

# Get current template
CURRENT_TEMPLATE=$(az containerapp show \
    --name "$CONTAINER_APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query 'properties.template' -o json)

# Update only the scale section
UPDATED_TEMPLATE=$(echo "$CURRENT_TEMPLATE" | jq \
  --arg polling "$POLLING_INTERVAL" \
  --arg cooldown "$COOLDOWN" \
  '.scale.pollingInterval = ($polling | tonumber) |
   .scale.cooldownPeriod = ($cooldown | tonumber)')

# Create minimal update payload
cat > /tmp/keda-update.json << EOF
{
  "properties": {
    "template": $UPDATED_TEMPLATE
  }
}
EOF

# Apply update
az rest \
    --method PATCH \
    --uri "/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RESOURCE_GROUP}/providers/Microsoft.App/containerApps/${CONTAINER_APP_NAME}?api-version=2023-11-02-preview" \
    --body @/tmp/keda-update.json

rm /tmp/keda-update.json

echo "✅ KEDA settings updated"
echo ""
echo "Verifying..."
az containerapp revision list \
    --name "$CONTAINER_APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query '[0].properties.template.scale.{pollingInterval, cooldownPeriod, minReplicas, maxReplicas}' -o json | jq '.'