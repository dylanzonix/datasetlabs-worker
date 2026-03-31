"""
Round-robin candidate dispatcher — feeds candidates to row generators.

V12: Processing is automatic. Candidates arrive in per-source buffers,
the dispatcher pulls them round-robin, and feeds them to row generators
via a shared semaphore (default 10 concurrent slots).

Backpressure: harvesters submit batches and wait for all candidates in
the batch to be processed before harvesting more. This prevents any
source from running up cost without validation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from dsl_worker.infra.candidate_pool import Candidate

logger = logging.getLogger(__name__)


# ── Data types ───────────────────────────────────────────────────────

@dataclass
class SourceResults:
    """Per-source outcome tracking."""
    candidates_total: int = 0
    processed: int = 0
    pending: int = 0
    rows: int = 0
    skipped: int = 0
    duplicates: int = 0
    errors: int = 0
    process_cost: float = 0.0
    skip_reasons: List[str] = field(default_factory=list)  # recent reasons, capped

    def snapshot(self) -> Dict[str, Any]:
        return {
            "candidates_total": self.candidates_total,
            "processed": self.processed,
            "pending": self.pending,
            "rows": self.rows,
            "skipped": self.skipped,
            "duplicates": self.duplicates,
            "errors": self.errors,
            "process_cost": self.process_cost,
            "skip_reasons": list(self.skip_reasons[-10:]),
        }


@dataclass
class BatchToken:
    """Tracks a batch of candidates for backpressure."""
    source_id: str
    count: int
    completed: int = 0
    event: asyncio.Event = field(default_factory=asyncio.Event)

    def mark_one_done(self):
        self.completed += 1
        if self.completed >= self.count:
            self.event.set()


@dataclass
class _PendingCandidate:
    """A candidate tagged with its source and batch for tracking."""
    candidate: Candidate
    source_id: str
    batch_token: BatchToken


# ── Dispatcher ───────────────────────────────────────────────────────

class CandidateDispatcher:
    """
    Round-robin dispatcher: pulls candidates from per-source buffers
    and feeds them to row generators.

    Usage:
        dispatcher = CandidateDispatcher(...)
        dispatcher_task = asyncio.create_task(dispatcher.run())

        # Harvesters push batches:
        token = dispatcher.submit_batch("harvest:0", candidates)
        await token.event.wait()  # backpressure

        # Orchestrator reads results:
        results = dispatcher.get_source_results("harvest:0")

        dispatcher.stop()
        await dispatcher_task
    """

    def __init__(
        self,
        generate_row_fn: Callable,
        semaphore: asyncio.Semaphore,
        generation_stats: Dict[str, Any],
        num_samples: int,
        stop_checker: Optional[Callable[[], bool]] = None,
        on_checkpoint: Optional[Callable] = None,
    ) -> None:
        self._generate_row_fn = generate_row_fn
        self._semaphore = semaphore
        self._generation_stats = generation_stats
        self._num_samples = num_samples
        self._stop_checker = stop_checker
        self._on_checkpoint = on_checkpoint

        # Per-source state
        self._source_queues: Dict[str, List[_PendingCandidate]] = {}
        self._source_results: Dict[str, SourceResults] = {}
        self._source_order: List[str] = []

        # Tracking
        self._active_tasks: set = set()
        self._stopped = False
        self._has_work = asyncio.Event()  # signaled when new candidates arrive
        self._pending_batch_tokens: List[BatchToken] = []

        # Delta tracking for dashboard "since last check-in"
        self._last_snapshot: Dict[str, Dict] = {}

    # ── Source management ─────────────────────────────────────────

    def add_source(self, source_id: str) -> None:
        """Register a new source. Call before submitting batches."""
        if source_id not in self._source_queues:
            self._source_queues[source_id] = []
            self._source_results[source_id] = SourceResults()
            self._source_order.append(source_id)
            logger.info(f"[dispatcher] Added source: {source_id}")

    def remove_source(self, source_id: str) -> None:
        """Stop pulling from a source. Remaining buffer still drains."""
        if source_id in self._source_order:
            self._source_order.remove(source_id)
            logger.info(f"[dispatcher] Removed source from rotation: {source_id}")

    def submit_batch(
        self, source_id: str, candidates: List[Candidate],
    ) -> BatchToken:
        """Submit a batch of candidates from a harvester.

        Returns a BatchToken whose .event is set when all candidates
        in this batch have been processed. Harvester awaits this for
        backpressure.
        """
        if source_id not in self._source_queues:
            self.add_source(source_id)

        token = BatchToken(source_id=source_id, count=len(candidates))
        if len(candidates) == 0:
            token.event.set()  # nothing to process
            return token

        results = self._source_results[source_id]
        results.candidates_total += len(candidates)
        results.pending += len(candidates)

        for candidate in candidates:
            self._source_queues[source_id].append(
                _PendingCandidate(
                    candidate=candidate,
                    source_id=source_id,
                    batch_token=token,
                )
            )

        self._pending_batch_tokens.append(token)
        self._has_work.set()
        return token

    # ── Results ───────────────────────────────────────────────────

    def get_source_results(self, source_id: str) -> SourceResults:
        """Get current results for a source."""
        return self._source_results.get(source_id, SourceResults())

    def get_all_results(self) -> Dict[str, SourceResults]:
        """Get results for all sources."""
        return dict(self._source_results)

    def get_results_delta(self) -> Dict[str, Dict]:
        """Get per-source result changes since last call to this method."""
        delta = {}
        for sid, results in self._source_results.items():
            current = results.snapshot()
            prev = self._last_snapshot.get(sid, {})
            d = {}
            for key in ("rows", "skipped", "duplicates", "errors", "process_cost", "processed"):
                d[key] = current.get(key, 0) - prev.get(key, 0)
            d["skip_reasons"] = results.skip_reasons[-5:]  # recent reasons
            if any(v for k, v in d.items() if k != "skip_reasons"):
                delta[sid] = d
        # Update snapshot
        self._last_snapshot = {
            sid: results.snapshot()
            for sid, results in self._source_results.items()
        }
        return delta

    @property
    def total_pending(self) -> int:
        return sum(len(q) for q in self._source_queues.values())

    @property
    def active_task_count(self) -> int:
        return len(self._active_tasks)

    @property
    def active_sources(self) -> List[str]:
        return list(self._source_order)

    # ── Main loop ─────────────────────────────────────────────────

    async def run(self) -> None:
        """Main dispatcher loop. Runs as a background task."""
        logger.info("[dispatcher] Started")
        rr_index = 0  # round-robin index

        while not self._stopped:
            # Check target
            rows_done = self._generation_stats.get("rows_generated", 0)
            if rows_done >= self._num_samples:
                logger.info("[dispatcher] Target reached, stopping")
                break

            if self._stop_checker and self._stop_checker():
                break

            # Clean up finished tasks
            done_tasks = {t for t in self._active_tasks if t.done()}
            for task in done_tasks:
                self._active_tasks.discard(task)
                self._handle_task_result(task)

            # Find next candidate via round-robin
            candidate_found = False
            if self._source_order:
                # Try each source starting from rr_index
                for _ in range(len(self._source_order)):
                    if rr_index >= len(self._source_order):
                        rr_index = 0
                    sid = self._source_order[rr_index]
                    rr_index += 1
                    queue = self._source_queues.get(sid, [])
                    if queue:
                        # Acquire semaphore before popping
                        try:
                            await asyncio.wait_for(
                                self._semaphore.acquire(), timeout=0.5,
                            )
                        except asyncio.TimeoutError:
                            # All slots full, wait a bit
                            break
                        except asyncio.CancelledError:
                            return

                        # Check stop again after await
                        if self._stopped:
                            self._semaphore.release()
                            return

                        # Pop and dispatch
                        pending = queue.pop(0)
                        task = asyncio.create_task(
                            self._process_one(pending)
                        )
                        self._active_tasks.add(task)
                        candidate_found = True
                        break  # back to top of while loop for round-robin fairness

            if not candidate_found:
                # Nothing to dispatch — wait for new work or check periodically
                self._has_work.clear()
                try:
                    await asyncio.wait_for(self._has_work.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass

        # Drain active tasks on shutdown
        await self._drain_active()
        logger.info("[dispatcher] Stopped")

    async def _process_one(self, pending: _PendingCandidate) -> None:
        """Process a single candidate. Semaphore already acquired."""
        try:
            result, cost, saved = await self._generate_row_fn(
                pending.candidate, pending.source_id,
            )
            self._record_result(pending, result, cost)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[dispatcher] Row gen error for {pending.source_id}: {e}")
            results = self._source_results.get(pending.source_id)
            if results:
                results.errors += 1
                results.processed += 1
                results.pending = max(0, results.pending - 1)
            pending.batch_token.mark_one_done()
        finally:
            self._semaphore.release()

    def _record_result(
        self, pending: _PendingCandidate, result: Any, cost: float,
    ) -> None:
        """Record a row generation outcome."""
        results = self._source_results.get(pending.source_id)
        if not results:
            return

        results.processed += 1
        results.pending = max(0, results.pending - 1)
        results.process_cost += cost

        if result.success:
            results.rows += 1
        elif result.skipped:
            if result.is_duplicate:
                results.duplicates += 1
            else:
                results.skipped += 1
                if result.skip_reason:
                    results.skip_reasons.append(result.skip_reason[:150])
                    # Cap stored reasons
                    if len(results.skip_reasons) > 20:
                        results.skip_reasons = results.skip_reasons[-10:]
        else:
            results.errors += 1

        pending.batch_token.mark_one_done()

    def _handle_task_result(self, task: asyncio.Task) -> None:
        """Handle a completed task (error logging only — results recorded inline)."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error(f"[dispatcher] Task error: {exc}")

    async def _drain_active(self) -> None:
        """Wait for all active tasks to complete on shutdown."""
        if not self._active_tasks:
            return
        logger.info(f"[dispatcher] Draining {len(self._active_tasks)} active tasks")
        done, _ = await asyncio.wait(self._active_tasks, timeout=30.0)
        for task in done:
            self._handle_task_result(task)
        # Cancel any stragglers
        for task in self._active_tasks - done:
            task.cancel()

    def stop(self) -> None:
        """Signal the dispatcher to stop."""
        self._stopped = True
        self._has_work.set()  # wake the loop
        # Release all pending batch tokens so harvesters don't deadlock
        for token in self._pending_batch_tokens:
            if not token.event.is_set():
                token.event.set()

    # ── Checkpoint support ────────────────────────────────────────

    def export_state(self) -> Dict[str, Any]:
        """Export dispatcher state for checkpointing."""
        return {
            "source_results": {
                sid: r.snapshot()
                for sid, r in self._source_results.items()
            },
            "pending_candidates": {
                sid: [
                    {
                        "values": p.candidate.values,
                        "source_id": p.candidate.source_id,
                        "source_context": p.candidate.source_context,
                        "metadata": p.candidate.metadata,
                    }
                    for p in queue
                ]
                for sid, queue in self._source_queues.items()
            },
        }

    def restore_results(self, state: Dict[str, Any]) -> None:
        """Restore result counters from checkpoint."""
        for sid, snapshot in state.get("source_results", {}).items():
            results = SourceResults(
                candidates_total=snapshot.get("candidates_total", 0),
                processed=snapshot.get("processed", 0),
                pending=snapshot.get("pending", 0),
                rows=snapshot.get("rows", 0),
                skipped=snapshot.get("skipped", 0),
                duplicates=snapshot.get("duplicates", 0),
                errors=snapshot.get("errors", 0),
                process_cost=snapshot.get("process_cost", 0.0),
                skip_reasons=snapshot.get("skip_reasons", []),
            )
            self._source_results[sid] = results
