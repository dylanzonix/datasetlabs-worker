"""
Sandboxed code execution using llm-sandbox.

Provides secure, isolated code execution with:
- Docker container isolation
- File mounting for uploaded/downloaded files
- Seed collection via stdout parsing
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Check if llm-sandbox is available
try:
    from llm_sandbox import SandboxSession
    from llm_sandbox.pool import PoolConfig, create_pool_manager
    SANDBOX_AVAILABLE = True
except ImportError:
    SANDBOX_AVAILABLE = False
    logger.warning("llm-sandbox not installed - code execution will be unsafe")


@dataclass
class SandboxResult:
    """Result from sandbox code execution."""
    success: bool
    stdout: str
    stderr: str
    seeds: List[Dict]  # Seeds extracted from execution
    error: Optional[str] = None


# Seed collection wrapper injected into scripts
SEED_WRAPPER_PREFIX = '''
import json as _json
import sys as _sys

# Seed collection
_collected_seeds = []

def add_seeds(seeds):
    """Add seeds to collection. Each seed should be a dict with text, note, source_url."""
    if isinstance(seeds, dict):
        seeds = [seeds]
    for s in seeds:
        _collected_seeds.append({
            "text": s.get("text"),
            "note": s.get("note", ""),
            "source_url": s.get("source_url"),
        })
    print(f"[add_seeds] Added {len(seeds)} seeds", file=_sys.stderr)

# Helper to read uploaded files
def read_file(filename):
    """Read an uploaded file from /sandbox/uploads/"""
    path = f"/sandbox/uploads/{filename}"
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {filename}. Available: {os.listdir('/sandbox/uploads/')}")
    with open(path, 'r') as f:
        return f.read()

def read_file_bytes(filename):
    """Read an uploaded file as bytes."""
    path = f"/sandbox/uploads/{filename}"
    with open(path, 'rb') as f:
        return f.read()

def list_files():
    """List available uploaded files."""
    return os.listdir('/sandbox/uploads/')

# Make page_markdown available if provided
page_markdown = open('/sandbox/page_markdown.txt').read() if os.path.exists('/sandbox/page_markdown.txt') else None
page_url = open('/sandbox/page_url.txt').read().strip() if os.path.exists('/sandbox/page_url.txt') else None

import os
import pandas as pd
from bs4 import BeautifulSoup
import re

###################
# USER SCRIPT START
###################
'''

SEED_WRAPPER_SUFFIX = '''
###################
# USER SCRIPT END
###################

# Output collected seeds as JSON
print("__SANDBOX_SEEDS_START__")
print(_json.dumps(_collected_seeds))
print("__SANDBOX_SEEDS_END__")
'''


class SandboxExecutor:
    """
    Execute code in an isolated sandbox.
    
    Uses llm-sandbox for Docker container isolation.
    Falls back to unsafe exec() if sandbox not available (development only).
    """
    
    def __init__(
        self,
        use_pool: bool = True,
        pool_size: int = 3,
        backend: str = "docker",
    ):
        self.use_pool = use_pool and SANDBOX_AVAILABLE
        self.backend = backend
        self._pool = None
        
        if self.use_pool:
            self._init_pool(pool_size)
    
    def _init_pool(self, pool_size: int):
        """Initialize container pool for better performance."""
        try:
            self._pool = create_pool_manager(
                backend=self.backend,
                config=PoolConfig(
                    max_pool_size=pool_size,
                    min_pool_size=1,
                    idle_timeout=300.0,  # 5 min
                    enable_prewarming=True,
                ),
                lang="python",
                libraries=["pandas", "beautifulsoup4", "openpyxl", "pdfplumber"],
            )
            logger.info(f"Sandbox pool initialized with {pool_size} containers")
        except Exception as e:
            logger.warning(f"Failed to initialize sandbox pool: {e}")
            self._pool = None
    
    def execute(
        self,
        script: str,
        page_markdown: Optional[str] = None,
        page_url: Optional[str] = None,
        uploaded_files: Optional[Dict[str, str]] = None,  # filename -> local_path
        timeout: int = 120,
    ) -> SandboxResult:
        """
        Execute code in sandbox.
        
        Args:
            script: Python code to execute
            page_markdown: Content of last fetched page
            page_url: URL of last fetched page
            uploaded_files: Dict mapping filename to local file path
            timeout: Execution timeout in seconds
            
        Returns:
            SandboxResult with stdout, stderr, and extracted seeds
        """
        if SANDBOX_AVAILABLE:
            return self._execute_sandboxed(
                script, page_markdown, page_url, uploaded_files, timeout
            )
        else:
            logger.warning("Running code UNSANDBOXED - this is unsafe!")
            return self._execute_unsafe(
                script, page_markdown, page_url, uploaded_files
            )
    
    def _execute_sandboxed(
        self,
        script: str,
        page_markdown: Optional[str],
        page_url: Optional[str],
        uploaded_files: Optional[Dict[str, str]],
        timeout: int,
    ) -> SandboxResult:
        """Execute in llm-sandbox container."""
        
        # Wrap script with seed collection
        wrapped_script = SEED_WRAPPER_PREFIX + script + SEED_WRAPPER_SUFFIX
        
        try:
            # Use pool if available, otherwise create session
            session_kwargs = {
                "lang": "python",
                "keep_template": True,
            }
            if self._pool:
                session_kwargs["pool"] = self._pool
            
            with SandboxSession(**session_kwargs) as session:
                # Create directories in sandbox
                session.execute_command("mkdir -p /sandbox/uploads /sandbox/downloads")
                
                # Copy page content if provided
                if page_markdown:
                    self._write_temp_and_copy(
                        session, page_markdown, "/sandbox/page_markdown.txt"
                    )
                if page_url:
                    self._write_temp_and_copy(
                        session, page_url, "/sandbox/page_url.txt"
                    )
                
                # Copy uploaded files
                if uploaded_files:
                    for filename, local_path in uploaded_files.items():
                        if os.path.exists(local_path):
                            session.copy_to_runtime(
                                local_path, f"/sandbox/uploads/{filename}"
                            )
                            logger.debug(f"Copied {filename} to sandbox")
                
                # Execute the wrapped script
                result = session.run(
                    wrapped_script,
                    libraries=["pandas", "beautifulsoup4", "openpyxl"],
                )
                
                # Parse seeds from output
                stdout = result.stdout or ""
                stderr = result.stderr or ""
                seeds = self._extract_seeds(stdout)
                
                # Remove seed markers from visible output
                clean_stdout = self._clean_output(stdout)
                
                return SandboxResult(
                    success=result.exit_code == 0,
                    stdout=clean_stdout,
                    stderr=stderr,
                    seeds=seeds,
                    error=stderr if result.exit_code != 0 else None,
                )
                
        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            return SandboxResult(
                success=False,
                stdout="",
                stderr=str(e),
                seeds=[],
                error=str(e),
            )
    
    def _write_temp_and_copy(self, session, content: str, dest_path: str):
        """Write content to temp file and copy to sandbox."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(content)
            temp_path = f.name
        try:
            session.copy_to_runtime(temp_path, dest_path)
        finally:
            os.unlink(temp_path)
    
    def _extract_seeds(self, stdout: str) -> List[Dict]:
        """Extract seeds from stdout."""
        seeds = []
        
        start_marker = "__SANDBOX_SEEDS_START__"
        end_marker = "__SANDBOX_SEEDS_END__"
        
        if start_marker in stdout and end_marker in stdout:
            try:
                start = stdout.index(start_marker) + len(start_marker)
                end = stdout.index(end_marker)
                seeds_json = stdout[start:end].strip()
                seeds = json.loads(seeds_json)
                logger.info(f"Extracted {len(seeds)} seeds from sandbox output")
            except (ValueError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to parse seeds from output: {e}")
        
        return seeds
    
    def _clean_output(self, stdout: str) -> str:
        """Remove seed markers from output."""
        start_marker = "__SANDBOX_SEEDS_START__"
        end_marker = "__SANDBOX_SEEDS_END__"
        
        if start_marker in stdout:
            # Remove everything from start marker to end
            idx = stdout.index(start_marker)
            return stdout[:idx].strip()
        
        return stdout
    
    def _execute_unsafe(
        self,
        script: str,
        page_markdown: Optional[str],
        page_url: Optional[str],
        uploaded_files: Optional[Dict[str, str]],
    ) -> SandboxResult:
        """
        Fallback: Execute with exec() - UNSAFE, for development only.
        """
        import io
        import sys
        
        collected_seeds = []
        
        def add_seeds(seeds):
            if isinstance(seeds, dict):
                seeds = [seeds]
            for s in seeds:
                collected_seeds.append({
                    "text": s.get("text"),
                    "note": s.get("note", ""),
                    "source_url": s.get("source_url"),
                })
        
        namespace = {
            "page_markdown": page_markdown,
            "page_url": page_url,
            "uploaded_files": uploaded_files or {},
            "add_seeds": add_seeds,
            "json": json,
            "re": __import__("re"),
            "os": os,
        }
        
        # Try to import common libs
        try:
            import pandas as pd
            namespace["pd"] = pd
            namespace["pandas"] = pd
        except ImportError:
            pass
        
        try:
            from bs4 import BeautifulSoup
            namespace["BeautifulSoup"] = BeautifulSoup
        except ImportError:
            pass
        
        # Capture stdout/stderr
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        try:
            exec(script, namespace)
            stdout = sys.stdout.getvalue()
            stderr = sys.stderr.getvalue()
            
            return SandboxResult(
                success=True,
                stdout=stdout,
                stderr=stderr,
                seeds=collected_seeds,
            )
        except Exception as e:
            stdout = sys.stdout.getvalue()
            stderr = sys.stderr.getvalue() + f"\n{e}"
            
            return SandboxResult(
                success=False,
                stdout=stdout,
                stderr=stderr,
                seeds=collected_seeds,
                error=str(e),
            )
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
    
    def close(self):
        """Cleanup resources."""
        if self._pool:
            try:
                self._pool.close()
                logger.info("Sandbox pool closed")
            except Exception as e:
                logger.warning(f"Error closing sandbox pool: {e}")