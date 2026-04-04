"""
Sandboxed code execution using sandbox_service.

Provides secure, isolated code execution with:
- Persistent sessions per agent (lazy-created on first use)
- Async HTTP-based execution via sandbox_service
- OOM detection, timeouts, memory limits
- Langfuse tracing for sandbox operations
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from sandbox_service import SandboxClient, SandboxSessionClient

logger = logging.getLogger(__name__)

# Langfuse is optional — tracing is a no-op if not configured
try:
    from langfuse import get_client as _get_langfuse_client

    def _get_langfuse():
        try:
            return _get_langfuse_client()
        except Exception:
            return None
except ImportError:
    def _get_langfuse():
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
        """Execute Python code, return result (with Langfuse span)."""
        langfuse = _get_langfuse()
        if langfuse:
            with langfuse.start_as_current_observation(
                as_type="span",
                name="sandbox:execute",
                input={"script_length": len(script), "timeout": timeout},
            ) as span:
                result = await self._do_execute(script, timeout)
                span.update(
                    output={
                        "success": result.success,
                        "exit_code": result.exit_code,
                        "oom_killed": result.oom_killed,
                        "timed_out": result.timed_out,
                    },
                    level="ERROR" if not result.success else "DEFAULT",
                )
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
        """Execute a shell command, return result (with Langfuse span)."""
        langfuse = _get_langfuse()
        if langfuse:
            with langfuse.start_as_current_observation(
                as_type="span",
                name="sandbox:exec_shell",
                input={"command_length": len(command), "timeout": timeout},
            ) as span:
                result = await self._do_exec_shell(command, timeout)
                span.update(
                    output={
                        "success": result.success,
                        "exit_code": result.exit_code,
                        "oom_killed": result.oom_killed,
                        "timed_out": result.timed_out,
                    },
                    level="ERROR" if not result.success else "DEFAULT",
                )
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
        langfuse = _get_langfuse()
        if langfuse:
            with langfuse.start_as_current_observation(
                as_type="span",
                name="sandbox:upload_workspace",
                input={
                    "file_url_count": len(file_urls) if file_urls else 0,
                },
            ):
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
        for subdir in ["downloads", "web", "extracted", "candidates"]:
            src = workspace_dir / subdir
            if src.exists() and any(src.iterdir()):
                try:
                    count = await self._session.upload_directory(src, dest_subdir=subdir)
                    logger.info(f"Uploaded {count} files from {subdir}/ to sandbox")
                except Exception as e:
                    logger.warning(f"Failed to upload {subdir}/ to sandbox: {e}")

        # Upload dsl_tools module so LLM can do: from dsl_tools import ...
        try:
            from dsl_worker.infra.dsl_tools_module import DSL_TOOLS_SOURCE
            await self._session.write_file("dsl_tools.py", DSL_TOOLS_SOURCE)
            logger.info("Uploaded dsl_tools.py to sandbox")
        except Exception as e:
            logger.warning(f"Failed to upload dsl_tools.py to sandbox: {e}")

    async def read_file(self, path: str) -> str:
        """Read a file from the sandbox workspace (with Langfuse span)."""
        langfuse = _get_langfuse()
        if langfuse:
            with langfuse.start_as_current_observation(
                as_type="span",
                name="sandbox:read_file",
                input={"path": path},
            ):
                return await self._session.read_file(path)
        return await self._session.read_file(path)

    async def close(self):
        """Destroy the session."""
        try:
            await self._client.destroy_session(self._session.session_id)
            logger.info(f"Sandbox session {self._session.session_id} destroyed")
        except Exception as e:
            logger.warning(f"Error destroying sandbox session: {e}")
