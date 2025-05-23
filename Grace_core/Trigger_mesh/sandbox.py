#!/usr/bin/env python3
"""
Sandbox Module for Grace AI System

This module defines a controlled execution environment ("sandbox") for Grace to simulate
or validate actions without affecting the live system. It provides capabilities for:
- Running healing actions, upgrades, or proposals in a simulated container
- Evaluating risk, impact, or resource changes before applying to production modules
- Logging predicted outcomes, warnings, or confidence levels for decision-making
- Supporting zero-downtime patching by testing actions in isolation
"""

import os
import sys
import json
import uuid
import time
import logging
import tempfile
import subprocess
import importlib
import inspect
import traceback
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
import shutil
import hashlib
import copy

# Mock imports for external dependencies
try:
    from cryptography.fernet import Fernet
except ImportError:
    # Mock Fernet for cryptography
    class Fernet:
        def __init__(self, key):
            self.key = key
            
        def encrypt(self, data):
            if isinstance(data, str):
                data = data.encode()
            return b"MOCK_ENCRYPTED_" + data
            
        def decrypt(self, token):
            if token.startswith(b"MOCK_ENCRYPTED_"):
                return token[len(b"MOCK_ENCRYPTED_"):]
            return token

try:
    from flask import Flask, request, jsonify
except ImportError:
    # Mock Flask
    class Flask:
        def __init__(self, name):
            self.name = name
            self.routes = {}
            
        def route(self, path, methods=None):
            def decorator(f):
                self.routes[path] = f
                return f
            return decorator
            
        def run(self, host=None, port=None, debug=False):
            print(f"Mock Flask app {self.name} running on {host}:{port}")
    
    class request:
        json = {}
        args = {}
        
    def jsonify(data):
        return json.dumps(data)

try:
    from werkzeug.security import generate_password_hash, check_password_hash
except ImportError:
    # Mock werkzeug security functions
    def generate_password_hash(password):
        return f"hashed_{password}"
        
    def check_password_hash(hash_value, password):
        return hash_value == f"hashed_{password}"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("grace.sandbox")

# Constants
SANDBOX_ROOT = os.environ.get("GRACE_SANDBOX_ROOT", "/tmp/grace_sandbox")
MAX_EXECUTION_TIME = int(os.environ.get("GRACE_SANDBOX_MAX_EXEC_TIME", "30"))  # seconds
DEFAULT_MEMORY_LIMIT = "512M"
DEFAULT_CPU_LIMIT = "0.5"

class SandboxException(Exception):
    """Base exception for all sandbox-related errors."""
    pass

class SandboxTimeoutError(SandboxException):
    """Raised when a sandbox execution exceeds the maximum allowed time."""
    pass

class SandboxResourceError(SandboxException):
    """Raised when a sandbox execution exceeds resource limits."""
    pass

class SandboxSecurityError(SandboxException):
    """Raised when a sandbox execution attempts a prohibited operation."""
    pass

class SandboxEnvironment:
    """
    Manages the creation, configuration, and cleanup of sandbox environments.
    """
    
    def __init__(self, 
                 sandbox_id: Optional[str] = None,
                 memory_limit: str = DEFAULT_MEMORY_LIMIT,
                 cpu_limit: str = DEFAULT_CPU_LIMIT,
                 network_enabled: bool = False,
                 persist: bool = False):
        """
        Initialize a new sandbox environment.
        
        Args:
            sandbox_id: Unique identifier for this sandbox. Generated if not provided.
            memory_limit: Memory limit for the sandbox (e.g., "512M", "1G").
            cpu_limit: CPU limit for the sandbox (e.g., "0.5", "1").
            network_enabled: Whether network access is allowed in the sandbox.
            persist: Whether to keep sandbox files after execution.
        """
        self.sandbox_id = sandbox_id or f"sandbox-{uuid.uuid4()}"
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.network_enabled = network_enabled
        self.persist = persist
        self.sandbox_path = os.path.join(SANDBOX_ROOT, self.sandbox_id)
        self.created_at = datetime.now()
        self.status = "initialized"
        self.results = {}
        
        # Create sandbox directory
        os.makedirs(self.sandbox_path, exist_ok=True)
        logger.info(f"Created sandbox environment at {self.sandbox_path}")
    
    def setup(self):
        """Prepare the sandbox environment for execution."""
        # Create necessary subdirectories
        os.makedirs(os.path.join(self.sandbox_path, "code"), exist_ok=True)
        os.makedirs(os.path.join(self.sandbox_path, "data"), exist_ok=True)
        os.makedirs(os.path.join(self.sandbox_path, "logs"), exist_ok=True)
        
        # Create metadata file
        metadata = {
            "sandbox_id": self.sandbox_id,
            "created_at": self.created_at.isoformat(),
            "memory_limit": self.memory_limit,
            "cpu_limit": self.cpu_limit,
            "network_enabled": self.network_enabled,
            "persist": self.persist
        }
        
        with open(os.path.join(self.sandbox_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        self.status = "ready"
        logger.info(f"Sandbox {self.sandbox_id} setup complete")
        return self
    
    def cleanup(self):
        """Clean up sandbox resources if not set to persist."""
        if not self.persist:
            logger.info(f"Cleaning up sandbox {self.sandbox_id}")
            try:
                shutil.rmtree(self.sandbox_path)
                logger.info(f"Removed sandbox directory {self.sandbox_path}")
            except Exception as e:
                logger.error(f"Failed to clean up sandbox {self.sandbox_id}: {str(e)}")
        else:
            logger.info(f"Sandbox {self.sandbox_id} persisted at {self.sandbox_path}")
    
    def add_file(self, source_path: str, destination_name: Optional[str] = None):
        """
        Copy a file into the sandbox environment.
        
        Args:
            source_path: Path to the source file
            destination_name: Name for the file in the sandbox (defaults to source filename)
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file {source_path} not found")
        
        dest_name = destination_name or os.path.basename(source_path)
        dest_path = os.path.join(self.sandbox_path, "code", dest_name)
        
        shutil.copy2(source_path, dest_path)
        logger.debug(f"Added file {source_path} to sandbox as {dest_name}")
        return dest_path
    
    def add_data(self, data: Union[str, bytes, Dict, List], filename: str):
        """
        Add data to the sandbox environment.
        
        Args:
            data: Data to write (string, bytes, or JSON-serializable object)
            filename: Name for the file in the sandbox
        """
        dest_path = os.path.join(self.sandbox_path, "data", filename)
        
        if isinstance(data, (dict, list)):
            with open(dest_path, "w") as f:
                json.dump(data, f, indent=2)
        elif isinstance(data, str):
            with open(dest_path, "w") as f:
                f.write(data)
        elif isinstance(data, bytes):
            with open(dest_path, "wb") as f:
                f.write(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        logger.debug(f"Added data to sandbox as {filename}")
        return dest_path
    
    def get_file(self, filename: str) -> str:
        """
        Get the contents of a file from the sandbox.
        
        Args:
            filename: Name of the file in the sandbox
            
        Returns:
            String contents of the file
        """
        file_path = os.path.join(self.sandbox_path, "code", filename)
        if not os.path.exists(file_path):
            file_path = os.path.join(self.sandbox_path, "data", filename)
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {filename} not found in sandbox")
            
        with open(file_path, "r") as f:
            return f.read()
    
    def get_binary_file(self, filename: str) -> bytes:
        """
        Get the binary contents of a file from the sandbox.
        
        Args:
            filename: Name of the file in the sandbox
            
        Returns:
            Binary contents of the file
        """
        file_path = os.path.join(self.sandbox_path, "code", filename)
        if not os.path.exists(file_path):
            file_path = os.path.join(self.sandbox_path, "data", filename)
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {filename} not found in sandbox")
            
        with open(file_path, "rb") as f:
            return f.read()

class SandboxExecutor:
    """
    Executes code within a sandbox environment with controlled resources and isolation.
    """
    
    def __init__(self, sandbox: SandboxEnvironment):
        """
        Initialize a sandbox executor.
        
        Args:
            sandbox: The sandbox environment to use for execution
        """
        self.sandbox = sandbox
        self.execution_timeout = MAX_EXECUTION_TIME
        self.results = {}
    
    def execute_python(self, code: str, filename: str = "sandbox_script.py") -> Dict[str, Any]:
        """
        Execute Python code in the sandbox.
        
        Args:
            code: Python code to execute
            filename: Name to save the code as
            
        Returns:
            Dictionary containing execution results
        """
        # Save the code to a file
        script_path = self.sandbox.add_data(code, filename)
        
        # Prepare the execution environment
        env = os.environ.copy()
        env["PYTHONPATH"] = self.sandbox.sandbox_path
        
        # Execute the code in a subprocess with timeout
        start_time = time.time()
        result = {
            "success": False,
            "output": "",
            "error": "",
            "execution_time": 0,
            "exit_code": None
        }
        
        try:
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=self.sandbox.sandbox_path
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.execution_timeout)
                result["output"] = stdout.decode("utf-8", errors="replace")
                result["error"] = stderr.decode("utf-8", errors="replace")
                result["exit_code"] = process.returncode
                result["success"] = process.returncode == 0
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                result["output"] = stdout.decode("utf-8", errors="replace")
                result["error"] = stderr.decode("utf-8", errors="replace") + "\nExecution timed out"
                result["exit_code"] = -1
                raise SandboxTimeoutError(f"Execution exceeded timeout of {self.execution_timeout}s")
                
        except Exception as e:
            if not isinstance(e, SandboxTimeoutError):
                result["error"] = f"Execution error: {str(e)}"
                result["exit_code"] = -1
                logger.error(f"Error executing Python code in sandbox: {str(e)}")
        
        finally:
            execution_time = time.time() - start_time
            result["execution_time"] = execution_time
            
            # Log the execution
            log_file = os.path.join(self.sandbox.sandbox_path, "logs", "execution.log")
            with open(log_file, "a") as f:
                f.write(f"--- Execution at {datetime.now().isoformat()} ---\n")
                f.write(f"Script: {filename}\n")
                f.write(f"Exit code: {result['exit_code']}\n")
                f.write(f"Execution time: {result['execution_time']:.2f}s\n")
                f.write("--- Output ---\n")
                f.write(result["output"])
                f.write("\n--- Errors ---\n")
                f.write(result["error"])
                f.write("\n\n")
        
        self.results = result
        return result
    
    def execute_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Execute a Python function in the sandbox.
        
        Args:
            func: Function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Dictionary containing execution results
        """
        # Get the function source code
        try:
            source = inspect.getsource(func)
        except (TypeError, OSError) as e:
            return {
                "success": False,
                "output": "",
                "error": f"Could not get source code for function: {str(e)}",
                "execution_time": 0,
                "exit_code": -1,
                "result": None
            }
        
        # Create a wrapper script that calls the function
        func_name = func.__name__
        module_name = func.__module__
        
        # Serialize arguments if possible
        try:
            args_str = json.dumps(args)
            kwargs_str = json.dumps(kwargs)
            serializable = True
        except (TypeError, OverflowError):
            serializable = False
        
        if serializable:
            wrapper_code = f"""
import json
import sys
import traceback
from {module_name} import {func_name}

try:
    args = json.loads('''{args_str}''')
    kwargs = json.loads('''{kwargs_str}''')
    result = {func_name}(*args, **kwargs)
    print("RESULT_START")
    print(json.dumps(result))
    print("RESULT_END")
    sys.exit(0)
except Exception as e:
    print("ERROR_START")
    traceback.print_exc()
    print("ERROR_END")
    sys.exit(1)
"""
        else:
            # For non-serializable arguments, we'll need to use the function directly
            # This is less secure but necessary for complex objects
            wrapper_code = f"""
import sys
import traceback
import pickle
import base64
from {module_name} import {func_name}

# The function and arguments are passed directly from the parent process
# This is less secure but necessary for non-serializable
