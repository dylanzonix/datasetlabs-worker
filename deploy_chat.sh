#!/usr/bin/env bash
# deploy_chat.sh — Build & deploy the chat worker FastAPI to Azure Container Apps
# with an HTTP-concurrent-requests scale rule and external ingress.
#
# Reuses the worker Dockerfile (same image, same dsl_api/dsl_worker code), but
# runs `entrypoint_chat.sh` instead of `entrypoint.sh`. No Service Bus involved.
#
# Expected files:
#   - Dockerfile in this directory
#   - .env in this directory (DATABASE_URL, AZURE_OPENAI_*, SUPABASE_JWT_SECRET, ...)
# Optional:
#   - deploy_chat.config in this directory

set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }
yaml_quote() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  printf "\"%s\"" "$s"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Peek for --production early so the right config file is loaded -----
PRODUCTION_MODE="0"
for arg in "$@"; do
  if [[ "$arg" == "--production" ]]; then
    PRODUCTION_MODE="1"
    break
  fi
done

# ---- Load deploy_chat.config ---------------------------------------------
if [[ "$PRODUCTION_MODE" == "1" ]] && [[ -f deploy_chat.config.prod ]]; then
  echo "Loading PRODUCTION configuration from deploy_chat.config.prod..."
  # shellcheck disable=SC1091
  source deploy_chat.config.prod
elif [[ -f deploy_chat.config ]]; then
  echo "Loading configuration from deploy_chat.config..."
  # shellcheck disable=SC1091
  source deploy_chat.config
fi

# ---- Defaults ------------------------------------------------------------
RESOURCE_GROUP="${RESOURCE_GROUP:-datasetlabs-rg}"
CONTAINER_APP_NAME="${CONTAINER_APP_NAME:-datasetlabs-chat-api}"
ENVIRONMENT_NAME="${ENVIRONMENT_NAME:-datasetlabs-env}"

REGISTRY_NAME="${REGISTRY_NAME:-datasetlabsregistry}"
IMAGE_NAME="${IMAGE_NAME:-worker}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FULL_IMAGE="${REGISTRY_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG}"

CPU="${CPU:-1.0}"
MEMORY="${MEMORY:-2.0Gi}"
MIN_REPLICAS="${MIN_REPLICAS:-1}"
MAX_REPLICAS="${MAX_REPLICAS:-10}"
CONCURRENT_REQUESTS="${CONCURRENT_REQUESTS:-5}"
TARGET_PORT="${TARGET_PORT:-8040}"
ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-http://localhost:8080}"

ENV_FILE="${ENV_FILE:-.env}"
# In --production mode default the env file to .env.prod unless the
# user explicitly passes --env-file. (This is set late so the
# deploy_chat.config.prod file's ENV_FILE export, if any, wins.)
if [[ "$PRODUCTION_MODE" == "1" && "$ENV_FILE" == ".env" ]]; then
  ENV_FILE=".env.prod"
fi

# ---- CLI args ------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --production) shift ;;  # already handled in the peek-ahead above
    --cpu) CPU="$2"; shift 2 ;;
    --memory) MEMORY="$2"; shift 2 ;;
    --min-replicas) MIN_REPLICAS="$2"; shift 2 ;;
    --max-replicas) MAX_REPLICAS="$2"; shift 2 ;;
    --image-tag) IMAGE_TAG="$2"; FULL_IMAGE="${REGISTRY_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG}"; shift 2 ;;
    --concurrent-requests) CONCURRENT_REQUESTS="$2"; shift 2 ;;
    --port) TARGET_PORT="$2"; shift 2 ;;
    --allowed-origins) ALLOWED_ORIGINS="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --help)
      cat <<EOF
Usage: $0 [options]

Container:
  --cpu VALUE
  --memory VALUE
  --min-replicas VALUE       (default: $MIN_REPLICAS)
  --max-replicas VALUE       (default: $MAX_REPLICAS)
  --image-tag VALUE

HTTP scaler:
  --concurrent-requests N    Scale up when concurrent reqs/replica > N (default: $CONCURRENT_REQUESTS)
  --port N                   Container port to expose (default: $TARGET_PORT)
  --allowed-origins CSV      CHAT_API_ALLOWED_ORIGINS for CORS (default: $ALLOWED_ORIGINS)

Env:
  --env-file PATH            (default: $ENV_FILE)
EOF
      exit 0
      ;;
    *) die "Unknown option: $1" ;;
  esac
done

# ---- Preflight -----------------------------------------------------------
need_cmd az
need_cmd docker

[[ -f "$ENV_FILE" ]] || die "$ENV_FILE not found in $SCRIPT_DIR"

az account show >/dev/null 2>&1 || die "Not logged into Azure. Run: az login"
SUBSCRIPTION_ID="$(az account show --query id -o tsv)"
[[ -n "$SUBSCRIPTION_ID" ]] || die "Could not determine subscription id"

if ! az containerapp -h >/dev/null 2>&1; then
  echo "az containerapp not found; installing containerapp extension..."
  az extension add -n containerapp >/dev/null
fi

# ---- Read env file -> YAML env list --------------------------------------
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
done < "$ENV_FILE"

# Append CORS + PORT
ENV_VARS_YAML+=$'        - name: CHAT_API_ALLOWED_ORIGINS\n'
ENV_VARS_YAML+=$'          value: '"$(yaml_quote "$ALLOWED_ORIGINS")"$'\n'
ENV_VARS_YAML+=$'        - name: PORT\n'
ENV_VARS_YAML+=$'          value: '"$(yaml_quote "$TARGET_PORT")"$'\n'

# ---- Print config --------------------------------------------------------
echo ""
echo "======================================"
echo "Chat Worker API Build & Deploy"
echo "======================================"
echo "Env File:          $ENV_FILE"
echo "Resource Group:    $RESOURCE_GROUP"
echo "App Name:          $CONTAINER_APP_NAME"
echo "Environment:       $ENVIRONMENT_NAME"
echo "Image:             $FULL_IMAGE"
echo "CPU / Memory:      $CPU / $MEMORY"
echo "Replicas:          min=$MIN_REPLICAS max=$MAX_REPLICAS"
echo "Target port:       $TARGET_PORT"
echo "Concurrent reqs:   $CONCURRENT_REQUESTS (per replica)"
echo "Allowed origins:   $ALLOWED_ORIGINS"
echo "======================================"
echo ""

# ---- Build + push --------------------------------------------------------
echo "Step 1/4: Logging into ACR..."
az acr login --name "$REGISTRY_NAME" >/dev/null

echo "Step 2/4: Building Docker image..."
# Build context is the parent dir so we can COPY api/ + sandbox/ + worker/.
docker build -f Dockerfile -t "$FULL_IMAGE" ..

echo "Step 3/4: Pushing Docker image..."
docker push "$FULL_IMAGE"

# ---- Create or update Container App --------------------------------------
echo "Step 4/4: Deploying Container App..."

APP_EXISTS="$(
  az containerapp show \
    -g "$RESOURCE_GROUP" -n "$CONTAINER_APP_NAME" \
    --query "name" -o tsv 2>/dev/null || true
)"

TMP_YAML="$(mktemp /tmp/${CONTAINER_APP_NAME}.XXXXXX.yaml)"
cat > "$TMP_YAML" <<EOF
properties:
  configuration:
    ingress:
      external: true
      targetPort: $TARGET_PORT
      transport: auto
      allowInsecure: false
  template:
    containers:
      - name: $CONTAINER_APP_NAME
        image: $FULL_IMAGE
        command: ["./entrypoint_chat.sh"]
        resources:
          cpu: $CPU
          memory: $MEMORY
        env:
$ENV_VARS_YAML
    scale:
      minReplicas: $MIN_REPLICAS
      maxReplicas: $MAX_REPLICAS
      rules:
        - name: http-scale
          http:
            metadata:
              concurrentRequests: "$CONCURRENT_REQUESTS"
EOF

if [[ -n "$APP_EXISTS" ]]; then
  az containerapp update \
    -g "$RESOURCE_GROUP" -n "$CONTAINER_APP_NAME" \
    --yaml "$TMP_YAML" >/dev/null
else
  REGISTRY_PASSWORD="$(az acr credential show --name "$REGISTRY_NAME" --query "passwords[0].value" -o tsv)"
  [[ -n "$REGISTRY_PASSWORD" ]] || die "Could not fetch ACR password (is ACR admin user enabled?)"

  # Create first without --yaml so registry creds are honored — `az
  # containerapp create --yaml ...` ignores all sibling flags including
  # --registry-*, which leaves the new app unable to pull from ACR.
  # Then update with the yaml to apply env vars + scaling. Same
  # workaround api/deploy.sh uses.
  az containerapp create \
    -g "$RESOURCE_GROUP" -n "$CONTAINER_APP_NAME" \
    --environment "$ENVIRONMENT_NAME" \
    --image "$FULL_IMAGE" \
    --registry-server "${REGISTRY_NAME}.azurecr.io" \
    --registry-username "$REGISTRY_NAME" \
    --registry-password "$REGISTRY_PASSWORD" \
    --ingress external \
    --target-port "$TARGET_PORT" \
    --min-replicas "$MIN_REPLICAS" \
    --max-replicas "$MAX_REPLICAS" \
    --cpu "$CPU" \
    --memory "$MEMORY" \
    >/dev/null

  az containerapp update \
    -g "$RESOURCE_GROUP" -n "$CONTAINER_APP_NAME" \
    --yaml "$TMP_YAML" >/dev/null
fi

rm -f "$TMP_YAML"

# Force new revision so env changes take effect
az containerapp update \
  -g "$RESOURCE_GROUP" -n "$CONTAINER_APP_NAME" \
  --set-env-vars FORCE_REDEPLOY="$(date +%s)" \
  --output none >/dev/null

FQDN="$(az containerapp show -g "$RESOURCE_GROUP" -n "$CONTAINER_APP_NAME" --query properties.configuration.ingress.fqdn -o tsv)"

echo ""
echo "======================================"
echo "✅ Chat worker deployed"
echo "======================================"
echo "URL: https://$FQDN"
echo ""
echo "Smoke test:"
echo "  curl -s https://$FQDN/v1/health"
echo ""
echo "App logs:"
echo "  az containerapp logs show -g $RESOURCE_GROUP -n $CONTAINER_APP_NAME --tail 200"
echo ""
