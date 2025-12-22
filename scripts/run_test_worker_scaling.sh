COUNT=100
URL=https://datasetlabs-api.grayisland-e1bbc667.eastus2.azurecontainerapps.io
TOKEN=$SUPABASE_TEST_JWT

python tests/test_worker_scaling.py $COUNT $URL $TOKEN