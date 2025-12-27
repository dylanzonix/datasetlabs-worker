#!/usr/bin/env bash
# deploy.sh — Build & deploy worker to Azure Container Apps + KEDA Service Bus scaling.
# - Builds/pushes Docker image to ACR
# - Creates/updates Container App with Service Bus scaler
# - Sets the scaler secret (Service Bus connection string)
# - Forces a new revision so secret changes take effect
# - Forces pollingInterval/cooldownPeriod via ARM PATCH (so they never show up as null)
# - Verifies scaling config via ARM GET
#
# NEW: Supports --noop flag to deploy a non-processing worker for local development
#
# Expected files:
#   - Dockerfile in this directory
#   - .env in this directory (contains AZURE_SERVICE_BUS_CONNECTION_STRING OR enough info to fetch it)
# Optional:
#   - deploy.config in this directory (bash variables, see defaults below)

set -euo pipefail

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
die() { echo "ERROR: $*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }

# Quote a value for YAML safely (very simple)
yaml_quote() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  printf "\"%s\"" "$s"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ------------------------------------------------------------------------------
# Load deploy.config (optional)
# ------------------------------------------------------------------------------
if [[ -f deploy.config ]]; then
  echo "Loading configuration from deploy.config..."
  # shellcheck disable=SC1091
  source deploy.config
fi

# ------------------------------------------------------------------------------
# Defaults (override via deploy.config or env vars)
# ------------------------------------------------------------------------------
RESOURCE_GROUP="${RESOURCE_GROUP:-datasetlabs-rg}"
CONTAINER_APP_NAME="${CONTAINER_APP_NAME:-datasetlabs-worker}"
ENVIRONMENT_NAME="${ENVIRONMENT_NAME:-datasetlabs-env}"

REGISTRY_NAME="${REGISTRY_NAME:-datasetlabsregistry}"
IMAGE_NAME="${IMAGE_NAME:-worker}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FULL_IMAGE="${REGISTRY_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG}"

CPU="${CPU:-2.0}"
MEMORY="${MEMORY:-4.0Gi}"
MIN_REPLICAS="${MIN_REPLICAS:-0}"
MAX_REPLICAS="${MAX_REPLICAS:-10}"

SERVICE_BUS_NAMESPACE="${SERVICE_BUS_NAMESPACE:-datasetlabs-bus}"
QUEUE_NAME="${QUEUE_NAME:-jobs}"
MESSAGE_COUNT="${MESSAGE_COUNT:-3}"

POLLING_INTERVAL="${POLLING_INTERVAL:-10}"
COOLDOWN_PERIOD="${COOLDOWN_PERIOD:-300}"

# Newer ARM api-version that supports pollingInterval/cooldownPeriod on Container Apps
ARM_API_VERSION="${ARM_API_VERSION:-2025-07-01}"

# If you want the script to auto-fetch the SB connection string when .env is wrong/missing:
AUTO_FETCH_SB_CONN="${AUTO_FETCH_SB_CONN:-1}"
SB_AUTH_RULE_NAME="${SB_AUTH_RULE_NAME:-RootManageSharedAccessKey}"

# NEW: Noop mode flag
NOOP_MODE="${NOOP_MODE:-0}"

# ------------------------------------------------------------------------------
# Parse CLI args (optional)
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) CPU="$2"; shift 2 ;;
    --memory) MEMORY="$2"; shift 2 ;;
    --min-replicas) MIN_REPLICAS="$2"; shift 2 ;;
    --max-replicas) MAX_REPLICAS="$2"; shift 2 ;;
    --image-tag) IMAGE_TAG="$2"; FULL_IMAGE="${REGISTRY_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG}"; shift 2 ;;
    --namespace) SERVICE_BUS_NAMESPACE="$2"; shift 2 ;;
    --queue) QUEUE_NAME="$2"; shift 2 ;;
    --message-count) MESSAGE_COUNT="$2"; shift 2 ;;
    --polling-interval) POLLING_INTERVAL="$2"; shift 2 ;;
    --cooldown) COOLDOWN_PERIOD="$2"; shift 2 ;;
    --arm-api-version) ARM_API_VERSION="$2"; shift 2 ;;
    --no-auto-fetch-sb-conn) AUTO_FETCH_SB_CONN="0"; shift 1 ;;
    --sb-auth-rule) SB_AUTH_RULE_NAME="$2"; shift 2 ;;
    --noop) NOOP_MODE="1"; shift 1 ;;
    --help)
      cat <<EOF
Usage: $0 [options]

Core:
  --cpu VALUE
  --memory VALUE
  --min-replicas VALUE
  --max-replicas VALUE
  --image-tag VALUE

Service Bus scaler:
  --namespace VALUE          (default: $SERVICE_BUS_NAMESPACE)
  --queue VALUE              (default: $QUEUE_NAME)
  --message-count VALUE      (default: $MESSAGE_COUNT)  # threshold for scaling

KEDA behavior:
  --polling-interval SECS    (default: $POLLING_INTERVAL)
  --cooldown SECS            (default: $COOLDOWN_PERIOD)

Development:
  --noop                     Deploy in noop mode (worker won't process messages)

Advanced:
  --arm-api-version VERSION  (default: $ARM_API_VERSION)
  --no-auto-fetch-sb-conn    disable auto-fetch of Service Bus conn string
  --sb-auth-rule NAME        (default: $SB_AUTH_RULE_NAME)
EOF
      exit 0
      ;;
    *) die "Unknown option: $1" ;;
  esac
done

# ------------------------------------------------------------------------------
# Preflight
# ------------------------------------------------------------------------------
need_cmd az
need_cmd docker

[[ -f .env ]] || die ".env not found in $SCRIPT_DIR"

az account show >/dev/null 2>&1 || die "Not logged into Azure. Run: az login"
SUBSCRIPTION_ID="$(az account show --query id -o tsv)"
[[ -n "$SUBSCRIPTION_ID" ]] || die "Could not determine subscription id"

# Ensure az containerapp is available (install extension only if needed)
if ! az containerapp -h >/dev/null 2>&1; then
  echo "az containerapp not found; installing containerapp extension..."
  az extension add -n containerapp >/dev/null
fi

# ------------------------------------------------------------------------------
# Read .env -> YAML env list AND load into environment for this shell
# ------------------------------------------------------------------------------
set -a
# shellcheck disable=SC1091
source .env
set +a

ENV_VARS_YAML=""
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" ]] && continue
  [[ "$line" =~ ^[[:space:]]*# ]] && continue

  key="${line%%=*}"
  value="${line#*=}"

  key="$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  [[ -z "$key" ]] && continue

  value="$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  value="$(echo "$value" | sed 's/^"//;s/"$//')"
  value="$(echo "$value" | sed "s/^'//;s/'$//")"

  ENV_VARS_YAML+=$'        - name: '"$key"$'\n'
  ENV_VARS_YAML+=$'          value: '"$(yaml_quote "$value")"$'\n'
done < .env

# NEW: Add WORKER_MODE env var if in noop mode
if [[ "$NOOP_MODE" == "1" ]]; then
  ENV_VARS_YAML+=$'        - name: WORKER_MODE\n'
  ENV_VARS_YAML+=$'          value: "noop"\n'
fi

# ------------------------------------------------------------------------------
# Service Bus connection string validation / fetch
# ------------------------------------------------------------------------------
SB_CONN="${AZURE_SERVICE_BUS_CONNECTION_STRING:-}"

has_sas_name=0
has_sas_key=0
if [[ "$SB_CONN" == *"SharedAccessKeyName="* ]]; then has_sas_name=1; fi
if [[ "$SB_CONN" == *"SharedAccessKey="* ]]; then has_sas_key=1; fi

if [[ -z "$SB_CONN" || $has_sas_name -eq 0 || $has_sas_key -eq 0 ]]; then
  if [[ "${AUTO_FETCH_SB_CONN}" == "1" ]]; then
    echo "AZURE_SERVICE_BUS_CONNECTION_STRING missing or malformed (needs SharedAccessKeyName + SharedAccessKey)."
    echo "Auto-fetching from Azure: namespace=$SERVICE_BUS_NAMESPACE authRule=$SB_AUTH_RULE_NAME ..."
    SB_CONN="$(
      az servicebus namespace authorization-rule keys list \
        -g "$RESOURCE_GROUP" \
        --namespace-name "$SERVICE_BUS_NAMESPACE" \
        --name "$SB_AUTH_RULE_NAME" \
        --query primaryConnectionString -o tsv
    )"
    [[ -n "$SB_CONN" ]] || die "Failed to fetch Service Bus connection string (check permissions / names)."
  else
    die "AZURE_SERVICE_BUS_CONNECTION_STRING missing or malformed. Put a full SAS conn string in .env."
  fi
fi

# ------------------------------------------------------------------------------
# Print config
# ------------------------------------------------------------------------------
echo ""
echo "======================================"
echo "Worker Build & Deploy"
if [[ "$NOOP_MODE" == "1" ]]; then
  echo "🔴 MODE: NOOP (worker will NOT process messages)"
else
  echo "MODE: NORMAL (worker will process messages)"
fi
echo "======================================"
echo "Resource Group:    $RESOURCE_GROUP"
echo "App Name:          $CONTAINER_APP_NAME"
echo "Environment:       $ENVIRONMENT_NAME"
echo "Image:             $FULL_IMAGE"
echo "CPU / Memory:      $CPU / $MEMORY"
echo "Replicas:          min=$MIN_REPLICAS max=$MAX_REPLICAS"
echo "Service Bus:       $SERVICE_BUS_NAMESPACE / $QUEUE_NAME"
echo "messageCount:      $MESSAGE_COUNT"
echo "pollingInterval:   ${POLLING_INTERVAL}s"
echo "cooldownPeriod:    ${COOLDOWN_PERIOD}s"
echo "ARM api-version:   $ARM_API_VERSION"
echo "======================================"
echo ""

# ------------------------------------------------------------------------------
# Step 1: Login to ACR
# ------------------------------------------------------------------------------
echo "Step 1/7: Logging into ACR..."
az acr login --name "$REGISTRY_NAME" >/dev/null

# ------------------------------------------------------------------------------
# Step 2: Build Docker image
# ------------------------------------------------------------------------------
echo "Step 2/7: Building Docker image..."
# Build from parent directory so we can access both api/ and worker/
# -f path is relative to current directory, not build context
docker build -f Dockerfile -t "$FULL_IMAGE" ..

# ------------------------------------------------------------------------------
# Step 3: Push Docker image
# ------------------------------------------------------------------------------
echo "Step 3/7: Pushing Docker image..."
docker push "$FULL_IMAGE"

# ------------------------------------------------------------------------------
# Step 4: Create or Update Container App (YAML)
# ------------------------------------------------------------------------------
echo "Step 4/7: Deploying Container App (create/update)..."

APP_EXISTS="$(
  az containerapp show \
    -g "$RESOURCE_GROUP" -n "$CONTAINER_APP_NAME" \
    --query "name" -o tsv 2>/dev/null || true
)"

TMP_YAML="$(mktemp /tmp/${CONTAINER_APP_NAME}.XXXXXX.yaml)"
cat > "$TMP_YAML" <<EOF
properties:
  template:
    containers:
      - name: $CONTAINER_APP_NAME
        image: $FULL_IMAGE
        resources:
          cpu: $CPU
          memory: $MEMORY
        env:
$ENV_VARS_YAML
    scale:
      minReplicas: $MIN_REPLICAS
      maxReplicas: $MAX_REPLICAS
      rules:
        - name: azure-servicebus-queue-rule
          custom:
            type: azure-servicebus
            metadata:
              queueName: $QUEUE_NAME
              namespace: $SERVICE_BUS_NAMESPACE
              messageCount: "$MESSAGE_COUNT"
            auth:
              - secretRef: servicebus-connection-string
                triggerParameter: connection
EOF

if [[ -n "$APP_EXISTS" ]]; then
  # Update existing app (this usually creates a new revision)
  az containerapp update \
    -g "$RESOURCE_GROUP" -n "$CONTAINER_APP_NAME" \
    --yaml "$TMP_YAML" >/dev/null
else
  # Create new app
  REGISTRY_PASSWORD="$(az acr credential show --name "$REGISTRY_NAME" --query "passwords[0].value" -o tsv)"
  [[ -n "$REGISTRY_PASSWORD" ]] || die "Could not fetch ACR password (is ACR admin user enabled?)"

  az containerapp create \
    -g "$RESOURCE_GROUP" -n "$CONTAINER_APP_NAME" \
    --environment "$ENVIRONMENT_NAME" \
    --image "$FULL_IMAGE" \
    --registry-server "${REGISTRY_NAME}.azurecr.io" \
    --registry-username "$REGISTRY_NAME" \
    --registry-password "$REGISTRY_PASSWORD" \
    --yaml "$TMP_YAML" >/dev/null
fi

rm -f "$TMP_YAML"

# ------------------------------------------------------------------------------
# Step 5: Set the Service Bus secret (for the scaler) + FORCE new revision
# ------------------------------------------------------------------------------
echo "Step 5/7: Setting scaler secret and forcing new revision..."

# Set secret (do NOT echo it)
az containerapp secret set \
  -g "$RESOURCE_GROUP" -n "$CONTAINER_APP_NAME" \
  --secrets "servicebus-connection-string=$SB_CONN" \
  --output none >/dev/null

# Force a new revision so updated secrets take effect
az containerapp update \
  -g "$RESOURCE_GROUP" -n "$CONTAINER_APP_NAME" \
  --set-env-vars FORCE_REDEPLOY="$(date +%s)" \
  --output none >/dev/null

# ------------------------------------------------------------------------------
# Step 6: Force pollingInterval/cooldownPeriod via ARM PATCH (avoid nulls)
# ------------------------------------------------------------------------------
echo "Step 6/7: Forcing pollingInterval/cooldownPeriod via ARM PATCH..."

APP_ID="/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RESOURCE_GROUP}/providers/Microsoft.App/containerApps/${CONTAINER_APP_NAME}"
APP_URL="https://management.azure.com${APP_ID}?api-version=${ARM_API_VERSION}"

az rest --method PATCH \
  --url "$APP_URL" \
  --body "{
    \"properties\": {
      \"template\": {
        \"scale\": {
          \"pollingInterval\": ${POLLING_INTERVAL},
          \"cooldownPeriod\": ${COOLDOWN_PERIOD}
        }
      }
    }
  }" >/dev/null

# ------------------------------------------------------------------------------
# Step 7: Verify via ARM GET + show replicas
# ------------------------------------------------------------------------------
echo "Step 7/7: Verifying scale config + replicas..."

echo ""
echo "Scale config (ARM):"
az rest --method GET --url "$APP_URL" --query "properties.template.scale" -o json

echo ""
REV="$(az containerapp show -g "$RESOURCE_GROUP" -n "$CONTAINER_APP_NAME" --query properties.latestRevisionName -o tsv)"
echo "Latest revision: $REV"
echo ""
echo "Replicas:"
az containerapp replica list -g "$RESOURCE_GROUP" -n "$CONTAINER_APP_NAME" --revision "$REV" -o table || true

echo ""
echo "======================================"
if [[ "$NOOP_MODE" == "1" ]]; then
  echo "✅ Worker deployed in NOOP mode!"
  echo "   Worker will NOT process messages from Service Bus"
  echo "   Local workers can now handle all messages for debugging"
else
  echo "✅ Worker deployed successfully!"
fi
echo "======================================"
echo ""
echo "Monitor:"
echo "  System logs:  az containerapp logs show -g $RESOURCE_GROUP -n $CONTAINER_APP_NAME --type system --tail 200"
echo "  App logs:     az containerapp logs show -g $RESOURCE_GROUP -n $CONTAINER_APP_NAME --tail 200"
echo "  Queue depth:  az servicebus queue show -g $RESOURCE_GROUP --namespace-name $SERVICE_BUS_NAMESPACE --name $QUEUE_NAME --query 'countDetails.activeMessageCount'"
echo ""