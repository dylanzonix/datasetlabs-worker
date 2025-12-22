#!/usr/bin/env python3
import requests
import sys
from datetime import datetime


def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <num_projects> <api_url> <jwt_token>")
        sys.exit(1)

    num_projects = int(sys.argv[1])
    api_url = sys.argv[2]
    jwt_token = sys.argv[3]

    # Generate run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {jwt_token}'
    }

    for i in range(1, num_projects + 1):
        # Create project
        create_data = {
            "name": f"Test Project {run_id}_{i}",
            "num_samples": 100,
            "generation_prompt": "Generate customer support questions"
        }

        print(f"Creating project {i}...")
        create_resp = requests.post(f"{api_url}/v1/projects", headers=headers, json=create_data)

        if create_resp.status_code != 200:
            print(f"Failed to create project {i}: {create_resp.status_code} - {create_resp.text}")
            continue

        project_id = create_resp.json()['id']
        print(f"Created project {i} with ID: {project_id}")

        # Start project
        start_resp = requests.post(f"{api_url}/v1/projects/{project_id}/start", headers=headers, json={})

        if start_resp.status_code != 200:
            print(f"Failed to start project {i}: {start_resp.status_code} - {start_resp.text}")
        else:
            print(f"Started project {i}")


if __name__ == "__main__":
    main()