az servicebus queue show \
  --resource-group datasetlabs-rg \
  --namespace-name datasetlabs-bus \
  --name jobs \
  --query "countDetails.activeMessageCount"
