az containerapp revision list \
  --name datasetlabs-worker \
  --resource-group datasetlabs-rg \
  --query "[0].{Name:name, Image:properties.template.containers[0].image, Created:properties.createdTime, Active:properties.active}" -o table
