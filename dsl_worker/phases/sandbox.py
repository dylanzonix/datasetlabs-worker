"""
Sandboxed code execution using llm-sandbox.

Provides secure, isolated code execution with:
- Docker container isolation
- Full workspace mounting
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

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


class SandboxExecutor:
    """Execute code in an isolated sandbox."""
    
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
        """Initialize container pool."""
        try:
            self._pool = create_pool_manager(
                backend=self.backend,
                config=PoolConfig(
                    max_pool_size=pool_size,
                    min_pool_size=1,
                    idle_timeout=300.0,
                    enable_prewarming=True,
                ),
                lang="python",
                libraries=["pandas", "beautifulsoup4", "openpyxl", "pdfplumber"],
            )
            logger.info(f"Sandbox pool initialized: {pool_size} containers")
        except Exception as e:
            logger.warning(f"Failed to init sandbox pool: {e}")
            self._pool = None
    
    def execute(
        self,
        script: str,
        workspace_dir: Optional[str] = None,
        timeout: int = 120,
    ) -> SandboxResult:
        """Execute code in sandbox."""
        if SANDBOX_AVAILABLE:
            return self._execute_sandboxed(script, workspace_dir, timeout)
        else:
            logger.warning("Running code UNSANDBOXED - development only!")
            return self._execute_unsafe(script, workspace_dir)
    
    def _execute_sandboxed(
        self,
        script: str,
        workspace_dir: Optional[str],
        timeout: int,
    ) -> SandboxResult:
        """Execute in llm-sandbox container."""
        
        wrapped_script = CODE_PREFIX + script + CODE_SUFFIX
        
        try:
            session_kwargs = {
                "lang": "python",
                "keep_template": True,
            }
            if self._pool:
                session_kwargs["pool"] = self._pool
            
            with SandboxSession(**session_kwargs) as session:
                if workspace_dir and os.path.exists(workspace_dir):
                    session.execute_command("mkdir -p /workspace")
                    
                    for subdir in ["uploads", "web", "extracted"]:
                        src_dir = os.path.join(workspace_dir, subdir)
                        if os.path.exists(src_dir):
                            session.execute_command(f"mkdir -p /workspace/{subdir}")
                            for filename in os.listdir(src_dir):
                                src_path = os.path.join(src_dir, filename)
                                if os.path.isfile(src_path):
                                    session.copy_to_runtime(
                                        src_path,
                                        f"/workspace/{subdir}/{filename}"
                                    )
                
                result = session.run(
                    wrapped_script,
                    libraries=["pandas", "beautifulsoup4", "openpyxl"],
                )
                
                stdout = result.stdout or ""
                stderr = result.stderr or ""
                
                return SandboxResult(
                    success=result.exit_code == 0,
                    stdout=stdout,
                    stderr=stderr,
                    error=stderr if result.exit_code != 0 else None,
                )
                
        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            return SandboxResult(
                success=False,
                stdout="",
                stderr=str(e),
                error=str(e),
            )
    
    def _execute_unsafe(
        self,
        script: str,
        workspace_dir: Optional[str],
    ) -> SandboxResult:
        """Fallback: Execute with exec() - UNSAFE."""
        import io
        import sys
        
        namespace = {
            "WORKSPACE": workspace_dir or ".",
            "json": json,
            "re": __import__("re"),
            "os": os,
        }
        
        def list_files(subdir=""):
            path = os.path.join(workspace_dir or ".", subdir) if subdir else (workspace_dir or ".")
            if os.path.exists(path):
                return os.listdir(path)
            return []
        
        def read_file(path):
            if not path.startswith("/"):
                path = os.path.join(workspace_dir or ".", path)
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        def write_file(path, content):
            if not path.startswith("/"):
                path = os.path.join(workspace_dir or ".", path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        namespace["list_files"] = list_files
        namespace["read_file"] = read_file
        namespace["write_file"] = write_file
        
        try:
            import pandas as pd
            namespace["pd"] = pd
        except ImportError:
            pass
        
        try:
            from bs4 import BeautifulSoup
            namespace["BeautifulSoup"] = BeautifulSoup
        except ImportError:
            pass
        
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
            )
        except Exception as e:
            stdout = sys.stdout.getvalue()
            stderr = sys.stderr.getvalue() + f"\n{e}"
            
            return SandboxResult(
                success=False,
                stdout=stdout,
                stderr=stderr,
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