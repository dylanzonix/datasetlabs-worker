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
    poll_timeout: int = 600,
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

    # Count lines and peek at first items for summary
    lines = [l for l in body.split("\n") if l.strip()]
    item_count = len(lines)
    field_names: List[str] = []
    sample_rows: List[Dict[str, Any]] = []
    try:
        first = json.loads(lines[0])
        field_names = sorted(first.keys())[:30]
    except (json.JSONDecodeError, IndexError):
        pass
    for line in lines[:3]:
        try:
            sample_rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return {
        "item_count": item_count,
        "file_path": str(output_path),
        "dataset_id": dataset_id,
        "fields": field_names,
        "sample_rows": sample_rows,
        "cost_usd": cost,
    }


async def _poll_run(
    run_id: str,
    headers: Dict[str, str],
    interval: int,
    timeout: int,
) -> tuple[Optional[str], float, Optional[str]]:
    """Poll an actor run until it finishes.

    Uses Apify's `waitForFinish=N` long-poll: the GET hangs up to N
    seconds and returns early as soon as the run reaches a terminal
    status. We loop in case the requested timeout exceeds the per-call
    cap (Apify caps waitForFinish at 60s).

    On our local timeout, we ABORT the run on Apify's side so the user
    doesn't keep getting billed for compute we'll never read.

    Returns: (dataset_id, cost_usd, error_message)
    """
    WAIT_FOR_FINISH_SECS = 60
    start = asyncio.get_event_loop().time()

    while True:
        elapsed = asyncio.get_event_loop().time() - start
        remaining = timeout - elapsed
        if remaining <= 0:
            break
        wait_secs = min(WAIT_FOR_FINISH_SECS, max(1, int(remaining)))
        try:
            async with httpx.AsyncClient(timeout=wait_secs + 10.0) as client:
                resp = await client.get(
                    f"{BASE_URL}/actor-runs/{run_id}",
                    headers=headers,
                    params={"waitForFinish": wait_secs},
                )
        except httpx.RequestError:
            await asyncio.sleep(1)
            continue

        if resp.status_code != 200:
            await asyncio.sleep(1)
            continue

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

    # Local timeout — abort the Apify run so it stops billing.
    await _abort_run(run_id, headers)
    return None, 0.0, (
        f"Actor run {run_id} did not finish within {timeout}s "
        f"(aborted on Apify side to stop billing). The actor may need "
        f"narrower input (smaller maxResults, fewer keywords) or a "
        f"longer timeout_secs on the next call."
    )


async def _abort_run(run_id: str, headers: Dict[str, str]) -> None:
    """POST /actor-runs/{id}/abort — stops the run on Apify's side.

    Best-effort: if it fails we just log; the run will eventually
    time out at Apify's max-runtime cap anyway.
    """
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{BASE_URL}/actor-runs/{run_id}/abort",
                headers=headers,
            )
        if resp.status_code in (200, 201):
            logger.info(f"[apify] Aborted run {run_id} after local timeout")
        else:
            logger.warning(
                f"[apify] Failed to abort run {run_id}: HTTP {resp.status_code}"
            )
    except Exception as e:
        logger.warning(f"[apify] abort run {run_id} raised: {e}")
