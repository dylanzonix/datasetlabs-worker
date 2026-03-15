"""
Sandboxed code execution using sandbox_service.

Provides secure, isolated code execution with:
- Persistent sessions per agent (lazy-created on first use)
- Async HTTP-based execution via sandbox_service
- OOM detection, timeouts, memory limits
- OTel tracing for sandbox operations
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from sandbox_service import SandboxClient, SandboxSessionClient

logger = logging.getLogger(__name__)

# OTel tracing is optional
try:
    from opentelemetry import trace as _otel_trace
    from openinference.semconv.trace import SpanAttributes as _SpanAttributes

    def _get_tracer():
        return _otel_trace.get_tracer(__name__)
except ImportError:
    def _get_tracer():
        return None


@dataclass
class SandboxResult:
    """Result from sandbox code execution."""
    success: bool
    stdout: str
    stderr: str
    exit_code: int = 0
    oom_killed: bool = False
    timed_out: bool = False
    error: Optional[str] = None


CODE_PREFIX = '''
import os
import sys
import json
import re

WORKSPACE = "/workspace"

def list_files(subdir=""):
    path = os.path.join(WORKSPACE, subdir) if subdir else WORKSPACE
    if os.path.exists(path):
        return os.listdir(path)
    return []

def read_file(path):
    if not path.startswith("/"):
        path = os.path.join(WORKSPACE, path)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def write_file(path, content):
    if not path.startswith("/"):
        path = os.path.join(WORKSPACE, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def grep(pattern, path, ignore_case=True):
    import re
    flags = re.IGNORECASE if ignore_case else 0
    content = read_file(path)
    matches = []
    for i, line in enumerate(content.split('\\n'), 1):
        if re.search(pattern, line, flags):
            matches.append(f"{i}: {line}")
    return "\\n".join(matches)

def head(path, n=50):
    content = read_file(path)
    lines = content.split('\\n')[:n]
    return "\\n".join(lines)

def tail(path, n=50):
    content = read_file(path)
    lines = content.split('\\n')[-n:]
    return "\\n".join(lines)

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

###################
# USER SCRIPT START
###################
'''

CODE_SUFFIX = '''
###################
# USER SCRIPT END
###################
'''


class SandboxSession:
    """Wraps a sandbox_service session for code execution."""

    def __init__(self, session_client: SandboxSessionClient, sandbox_client: SandboxClient):
        self._session = session_client
        self._client = sandbox_client

    @property
    def session_id(self) -> str:
        return self._session.session_id

    async def execute(self, script: str, timeout: int = 120) -> SandboxResult:
        """Execute Python code, return result (with OTel span)."""
        tracer = _get_tracer()
        if tracer:
            with tracer.start_as_current_span(
                "sandbox:execute",
                attributes={_SpanAttributes.OPENINFERENCE_SPAN_KIND: "TOOL"},
            ) as span:
                span.set_attribute("input.value", str({"script_length": len(script), "timeout": timeout})[:500])
                result = await self._do_execute(script, timeout)
                span.set_attribute("output.value", str({
                    "success": result.success,
                    "exit_code": result.exit_code,
                    "oom_killed": result.oom_killed,
                    "timed_out": result.timed_out,
                })[:500])
                return result
        return await self._do_execute(script, timeout)

    async def _do_execute(self, script: str, timeout: int) -> SandboxResult:
        """Execute Python code, return result."""
        wrapped = CODE_PREFIX + script + CODE_SUFFIX

        try:
            result = await self._session.exec_python(wrapped, timeout=timeout)
            return SandboxResult(
                success=result.success and result.exit_code == 0,
                stdout=result.stdout or "",
                stderr=result.stderr or "",
                exit_code=result.exit_code,
                oom_killed=result.oom_killed,
                timed_out=result.timed_out,
                error=(result.stderr or None) if result.exit_code != 0 else None,
            )
        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            return SandboxResult(
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                error=str(e),
            )

    async def exec_shell(self, command: str, timeout: int = 60) -> SandboxResult:
        """Execute a shell command, return result (with OTel span)."""
        tracer = _get_tracer()
        if tracer:
            with tracer.start_as_current_span(
                "sandbox:exec_shell",
                attributes={_SpanAttributes.OPENINFERENCE_SPAN_KIND: "TOOL"},
            ) as span:
                span.set_attribute("input.value", str({"command_length": len(command), "timeout": timeout})[:500])
                result = await self._do_exec_shell(command, timeout)
                span.set_attribute("output.value", str({
                    "success": result.success,
                    "exit_code": result.exit_code,
                    "oom_killed": result.oom_killed,
                    "timed_out": result.timed_out,
                })[:500])
                return result
        return await self._do_exec_shell(command, timeout)

    async def _do_exec_shell(self, command: str, timeout: int) -> SandboxResult:
        """Execute a shell command, return result."""
        try:
            result = await self._session.exec_shell(command, timeout=timeout)
            return SandboxResult(
                success=result.success and result.exit_code == 0,
                stdout=result.stdout or "",
                stderr=result.stderr or "",
                exit_code=result.exit_code,
                oom_killed=result.oom_killed,
                timed_out=result.timed_out,
                error=(result.stderr or None) if result.exit_code != 0 else None,
            )
        except Exception as e:
            logger.error(f"Sandbox shell execution failed: {e}")
            return SandboxResult(
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                error=str(e),
            )

    async def upload_workspace(
        self,
        workspace_dir: Path,
        file_urls: Optional[Dict[str, str]] = None,
    ):
        """Upload workspace files to sandbox.

        For uploaded files: uses file_urls (sandbox service fetches from SAS URLs).
        For other dirs (downloads/, web/, extracted/): uploads from local disk.
        """
        tracer = _get_tracer()
        if tracer:
            with tracer.start_as_current_span(
                "sandbox:upload_workspace",
                attributes={_SpanAttributes.OPENINFERENCE_SPAN_KIND: "TOOL"},
            ) as span:
                span.set_attribute("input.value", str({"file_url_count": len(file_urls) if file_urls else 0})[:500])
                await self._do_upload_workspace(workspace_dir, file_urls)
        else:
            await self._do_upload_workspace(workspace_dir, file_urls)

    async def _do_upload_workspace(
        self,
        workspace_dir: Path,
        file_urls: Optional[Dict[str, str]] = None,
    ):
        """Upload workspace files to sandbox (implementation)."""
        # Fetch uploaded files via sandbox service (no local disk needed)
        if file_urls:
            for filename, url in file_urls.items():
                try:
                    await self._session.fetch_from_url(url, f"uploads/{filename}")
                    logger.info(f"Sandbox fetched upload: {filename}")
                except Exception as e:
                    logger.warning(f"Failed to fetch {filename} into sandbox: {e}")

        # Upload local dirs (downloads from browser, etc.)
        for subdir in ["downloads", "web", "extracted"]:
            src = workspace_dir / subdir
            if src.exists() and any(src.iterdir()):
                try:
                    count = await self._session.upload_directory(src, dest_subdir=subdir)
                    logger.info(f"Uploaded {count} files from {subdir}/ to sandbox")
                except Exception as e:
                    logger.warning(f"Failed to upload {subdir}/ to sandbox: {e}")

    async def read_file(self, path: str) -> str:
        """Read a file from the sandbox workspace (with OTel span)."""
        tracer = _get_tracer()
        if tracer:
            with tracer.start_as_current_span(
                "sandbox:read_file",
                attributes={_SpanAttributes.OPENINFERENCE_SPAN_KIND: "TOOL"},
            ) as span:
                span.set_attribute("input.value", str({"path": path})[:500])
                return await self._session.read_file(path)
        return await self._session.read_file(path)

    async def close(self):
        """Destroy the session."""
        try:
            await self._client.destroy_session(self._session.session_id)
            logger.info(f"Sandbox session {self._session.session_id} destroyed")
        except Exception as e:
            logger.warning(f"Error destroying sandbox session: {e}")
