"""
Download Apify datasets to local JSONL files.

Bridge between MCP-based actor runs (which return a runId or datasetId)
and our file-based candidate pipeline.

Supports two modes:
- dataset_id: download immediately
- run_id: poll until actor finishes, then download
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.apify.com/v2"


async def download_apify_results(
    api_key: str,
    output_path: Path,
    dataset_id: Optional[str] = None,
    run_id: Optional[str] = None,
    limit: Optional[int] = None,
    fields: Optional[List[str]] = None,
    poll_interval: int = 5,
    poll_timeout: int = 300,
) -> Dict[str, Any]:
    """Download Apify actor results to a JSONL file.

    Accepts either dataset_id (immediate download) or run_id (polls
    until the actor finishes, then downloads from its dataset).

    Args:
        api_key: Apify API token
        output_path: Where to write the JSONL file
        dataset_id: Dataset ID for immediate download
        run_id: Run ID to poll for completion first
        limit: Max items to download (None = all)
        fields: Only include these fields (None = all)
        poll_interval: Seconds between status checks when polling
        poll_timeout: Max seconds to wait for actor to finish

    Returns:
        dict with: item_count, file_path, fields, sample, cost_usd
    """
    if not dataset_id and not run_id:
        return {"error": "Provide either dataset_id or run_id", "item_count": 0}

    headers = {"Authorization": f"Bearer {api_key}"}

    # If we have a run_id, poll until finished and get the dataset_id
    if run_id and not dataset_id:
        dataset_id, cost, error = await _poll_run(
            run_id, headers, poll_interval, poll_timeout
        )
        if error:
            return {"error": error, "item_count": 0, "run_id": run_id}
    else:
        cost = 0.0

    # Download dataset items as JSONL — stream straight to disk
    params: Dict[str, Any] = {"format": "jsonl"}
    if limit:
        params["limit"] = limit
    if fields:
        params["fields"] = ",".join(fields)

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.get(
            f"{BASE_URL}/datasets/{dataset_id}/items",
            params=params,
            headers=headers,
        )

    if resp.status_code != 200:
        return {
            "error": f"Failed to download dataset {dataset_id}: HTTP {resp.status_code}",
            "item_count": 0,
        }

    body = resp.text.strip()
    if not body:
        return {
            "item_count": 0,
            "file_path": str(output_path),
            "dataset_id": dataset_id,
        }

    # Write raw JSONL directly — no parsing/re-serialization
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(body + "\n", encoding="utf-8")

    # Count lines and peek at first item for summary
    lines = body.split("\n")
    item_count = len([l for l in lines if l.strip()])
    field_names: List[str] = []
    try:
        first = json.loads(lines[0])
        field_names = sorted(first.keys())[:15]
    except (json.JSONDecodeError, IndexError):
        pass

    return {
        "item_count": item_count,
        "file_path": str(output_path),
        "dataset_id": dataset_id,
        "fields": field_names,
        "cost_usd": cost,
    }


async def _poll_run(
    run_id: str,
    headers: Dict[str, str],
    interval: int,
    timeout: int,
) -> tuple[Optional[str], float, Optional[str]]:
    """Poll an actor run until it finishes.

    Returns: (dataset_id, cost_usd, error_message)
    """
    elapsed = 0
    while elapsed < timeout:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{BASE_URL}/actor-runs/{run_id}",
                headers=headers,
            )

        if resp.status_code != 200:
            return None, 0.0, f"Failed to check run {run_id}: HTTP {resp.status_code}"

        data = resp.json().get("data", {})
        status = data.get("status", "RUNNING")

        if status == "SUCCEEDED":
            dataset_id = data.get("defaultDatasetId")
            cost = data.get("usageTotalUsd", 0.0) or 0.0
            if not dataset_id:
                return None, cost, "Run succeeded but no dataset ID returned"
            logger.info(
                f"[apify] Run {run_id} succeeded: dataset={dataset_id}, cost=${cost:.4f}"
            )
            return dataset_id, cost, None

        if status in ("FAILED", "ABORTED", "TIMED-OUT"):
            return None, 0.0, f"Actor run {status}: {run_id}"

        await asyncio.sleep(interval)
        elapsed += interval

    return None, 0.0, f"Timed out waiting for run {run_id} after {timeout}s"
