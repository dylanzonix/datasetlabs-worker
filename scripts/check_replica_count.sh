az containerapp revision list \
  --name datasetlabs-worker \
  --resource-group datasetlabs-rg \
  --query "[0].properties.replicas" -o table