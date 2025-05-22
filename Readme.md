# Grace - Autonomous Code Assistant

Grace is an evolving system builder that can parse, run, and evaluate code safely, learn from its execution, and export complete projects.

## Components

1. **Code Execution Kernel** - A secure engine that can parse, run, and evaluate Python or other supported languages safely.
2. **Self-Learning Loop** - Logic that logs output, evaluates errors, attempts fixes, and re-runs, creating a closed feedback loop.
3. **Universal Export Layer** - Ability to output full projects to GitHub, zip files, Docker containers, or Vercel.
4. **Teach-and-Train Interface** - Understanding of commands like "teach X" or "train on Y" to parse files and integrate logic.
5. **Trigger Mesh (Watchdog)** - Handles condition-based actions, triggering self-repair, contributor pings, or rollback.
6. **API Output Mode** - HTTP interface or CLI where users can submit goals and the agent will auto-architect, scaffold, and implement.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m grace.main --goal "build me a parser"
```

## Configuration

Edit `config.yaml` to customize Grace's behavior.
```

Now let's create the requirements file:

```text:requirements.txt
pyyaml>=6.0
fastapi>=0.95.0
uvicorn>=0.21.1
requests>=2.28.2
gitpython>=3.1.31
docker>=6.0.1
watchdog>=3.0.0
pytest>=7.3.1
mypy>=1.2.0
black>=23.3.0
```

Let's create the main module structure:

```python:grace/__init__.py
"""Grace - Autonomous Code Assistant."""

__version__ = "0.1.0"
```

Now let's implement the main configuration:

```python:grace/config.py
"""Configuration management for Grace."""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("grace.log"),
    ],
)

logger = logging.getLogger("grace")

DEFAULT_CONFIG = {
    "execution": {
        "timeout": 30,  # seconds
        "max_memory": 512,  # MB
        "allowed_modules": ["os", "sys", "pathlib", "re", "json", "yaml", "math"],
        "sandbox_mode": "strict",
    },
    "learning": {
        "max_attempts": 5,
        "error_threshold": 0.8,
        "success_threshold": 0.9,
        "learning_rate": 0.1,
    },
    "export": {
        "github_token_env": "GITHUB_TOKEN",
        "default_repo_name": "grace-project",
        "default_docker_tag": "grace-latest",
    },
    "training": {
        "data_dir": "training_data",
        "model_dir": "models",
        "batch_size": 32,
    },
    "watchdog": {
        "check_interval": 5,  # seconds
        "watched_paths": ["src", "tests"],
        "auto_repair": True,
    },
    "api": {
        "host": "127.0.0.1",
        "port": 8000,
        "debug": False,
    },
}


class Config:
    """Configuration manager for Grace."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config_path = config_path
        self.config = DEFAULT_CONFIG.copy()
        
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        else:
            logger.warning(f"No config file found at {config_path}, using defaults")
            
    def _load_config(self, config_path: str) -> None:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, "r") as f:
                user_config = yaml.safe_load(f)
                
            # Update config with user values
            for section, values in user_config.items():
                if section in self.config:
                    if isinstance(values, dict):
                        self.config[section].update(values)
                    else:
                        self.config[section] = values
                else:
                    self.config[section] = values
                    
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {str(e)}")
            
    def get(self, section: str, key: Optional[str] = None) -> Any:
        """Get configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key within section. If None, returns entire section.
            
        Returns:
            Configuration value or section
        """
        if section not in self.config:
            logger.warning(f"Config section {section} not found, using empty dict")
            return {} if key is None else None
            
        if key is None:
            return self.config[section]
            
        if key not in self.config[section]:
            logger.warning(f"Config key {key} not found in section {section}")
            return None
            
        return self.config[section][key]
        
    def save(self, config_path: Optional[str] = None) -> None:
        """Save current configuration to file.
        
        Args:
            config_path: Path to save configuration. If None, uses self.config_path.
        """
        save_path = config_path or self.config_path
        
        if not save_path:
            logger.error("No config path specified for saving")
            return
            
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            with open(save_path, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
                
            logger.info(f"Saved configuration to {save_path}")
        except Exception as e:
            logger.error(f"Error saving config to {save_path}: {str(e)}")
```

Now let's implement the Code Execution Kernel:

```python:grace/execution.py
"""Code Execution Kernel for Grace.

This module provides a secure sandbox for executing code and capturing results.
"""

import os
import sys
import subprocess
import tempfile
import logging
import resource
import signal
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import contextlib
import io
import traceback

from grace.config import Config

logger = logging.getLogger("grace.execution")


class ExecutionResult:
    """Result of code execution."""

    def __init__(
        self,
        success: bool,
        output: str,
        error: str,
        execution_time: float,
        memory_usage: float,
        return_value: Any = None,
    ):
        """Initialize execution result.
        
        Args:
            success: Whether execution was successful
            output: Standard output from execution
            error: Standard error from execution
            execution_time: Time taken for execution (seconds)
            memory_usage: Memory used during execution (MB)
            return_value: Return value from execution (if applicable)
        """
        self.success = success
        self.output = output
        self.error = error
        self.execution_time = execution_time
        self.memory_usage = memory_usage
        self.return_value = return_value
        
    def __str__(self) -> str:
        """String representation of execution result."""
        status = "SUCCESS" if self.success else "FAILURE"
        return (
            f"Execution {status}\n"
            f"Time: {self.execution_time:.2f}s\n"
            f"Memory: {self.memory_usage:.2f}MB\n"
            f"Output: {self.output[:100]}{'...' if len(self.output) > 100 else ''}\n"
            f"Error: {self.error[:100]}{'...' if len(self.error) > 100 else ''}"
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "return_value": str(self.return_value) if self.return_value is not None else None,
        }


class CodeExecutionKernel:
    """Secure engine for executing code in a sandbox."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize code execution kernel.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or Config()
        self.timeout = self.config.get("execution", "timeout")
        self.max_memory = self.config.get("execution", "max_memory")
        self.allowed_modules = self.config.get("execution", "allowed_modules")
        self.sandbox_mode = self.config.get("execution", "sandbox_mode")
        
    def _limit_resources(self) -> None:
        """Set resource limits for subprocess."""
        # Set memory limit (in bytes)
        memory_bytes = self.max_memory * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        
    def execute_python_subprocess(self, code: str) -> ExecutionResult:
        """Execute Python code in a subprocess.
        
        Args:
            code: Python code to execute
            
        Returns:
            ExecutionResult object with execution details
        """
        # Create temporary file with code
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code)
            temp_file = f.name
            
        try:
            # Prepare command with resource limitations
            cmd = [sys.executable, temp_file]
            
            # Start process with timeout
            start_time = os.times()
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=self._limit_resources,
                text=True,
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                end_time = os.times()
                
                # Calculate execution time (user + system time)
                execution_time = (
                    (end_time.user - start_time.user) + 
                    (end_time.system - start_time.system)
                )
                
                # Estimate memory usage (this is approximate)
                # In a real implementation, you'd want to use psutil or similar
                memory_usage = self.max_memory * 0.5  # Placeholder
                
                success = process.returncode == 0
                
                return ExecutionResult(
                    success=success,
                    output=stdout,
                    error=stderr,
                    execution_time=execution_time,
                    memory_usage=memory_usage,
                )
                
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                return ExecutionResult(
                    success=False,
                    output=stdout,
                    error=f"Execution timed out after {self.timeout} seconds",
                    execution_time=self.timeout,
                    memory_usage=self.max_memory,  # Assume max usage on timeout
                )
                
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution error: {str(e)}",
                execution_time=0.0,
                memory_usage=0.0,
            )
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except:
                pass
                
    def execute_python_inline(self, code: str) -> ExecutionResult:
        """Execute Python code in the current process with restrictions.
        
        This is less secure than subprocess execution but faster for simple code.
        
        Args:
            code: Python code to execute
            
        Returns:
            ExecutionResult object with execution details
        """
        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Track execution time and success
        start_time = os.times()
        success = True
        return_value = None
        
        # Set up timeout handler
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Execution timed out after {self.timeout} seconds")
            
        try:
            # Set timeout alarm
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)
            
            # Execute code with captured output
            with contextlib.redirect_stdout(stdout_capture):
                with contextlib.redirect_stderr(stderr_capture):
                    # Compile code to catch syntax errors before execution
                    compiled_code = compile(code, "<string>", "exec")
                    
                    # Create restricted globals
                    restricted_globals = {
                        "__builtins__": {
                            name: getattr(__builtins__, name)
                            for name in dir(__builtins__)
                            if name not in ["__import__", "eval", "exec", "open"]
                        }
                    }
                    
                    # Add allowed modules
                    for module_name in self.allowed_modules:
                        try:
                            restricted_globals[module_name] = __import__(module_name)
                        except ImportError:
                            pass
                            
                    # Execute code
                    local_vars = {}
                    exec(compiled_code, restricted_globals, local_vars)
                    
                    # Check for return value
                    if "result" in local_vars:
                        return_value = local_vars["result"]
                        
        except Exception as e:
            success = False
            stderr_capture.write(f"Exception: {str(e)}\n")
            stderr_capture.write(traceback.format_exc())
            
        finally:
            # Cancel timeout alarm
            signal.alarm(0)
            
        # Calculate execution time
        end_time = os.times()
        execution_time = (
            (end_time.user - start_time.user) + 
            (end_time.system - start_time.system)
        )
        
        # Estimate memory usage (placeholder)
        memory_usage = 0.0  # In a real implementation, use memory_profiler or similar
        
        return ExecutionResult(
