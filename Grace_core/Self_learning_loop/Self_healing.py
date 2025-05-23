import os
from pathlib import Path
import yaml
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

@dataclass
class WatcherConfig:
    enabled: bool
    interval_seconds: int
    thresholds: Dict[str, float]
    triggers: List[str]

@dataclass
class DiagnosticConfig:
    scan_depth: str  # "light", "medium", "deep"
    timeout_seconds: int
    components: List[str]

@dataclass
class HealingConfig:
    auto_heal: bool
    strategies: List[str]
    max_attempts: int
    cooldown_seconds: int

@dataclass
class MemoryConfig:
    working_memory_size: int
    persistent_storage_path: str
    retention_days: int
    encryption_enabled: bool

@dataclass
class LoggingConfig:
    level: str
    file_path: str
    rotation_size_mb: int
    retention_count: int

@dataclass
class Config:
    watchers: Dict[str, WatcherConfig]
    diagnostics: DiagnosticConfig
    healing: HealingConfig
    memory: MemoryConfig
    logging: LoggingConfig
    trust_threshold: float
    telemetry_enabled: bool
    environment: str

class ConfigManager:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load configuration from YAML file or environment variables"""
        config_path = os.environ.get('SELF_HEALING_CONFIG', 'config.yaml')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            # Default configuration
            config_data = self._get_default_config()
            
        # Override with environment variables if present
        self._override_from_env(config_data)
        
        # Create config objects
        self._config = self._create_config_objects(config_data)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            "watchers": {
                "stress_loop": {
                    "enabled": True,
                    "interval_seconds": 60,
                    "thresholds": {
                        "cpu_percent": 80.0,
                        "memory_percent": 85.0,
                        "disk_percent": 90.0
                    },
                    "triggers": ["system_event", "manual"]
                },
                "kpi_tracker": {
                    "enabled": True,
                    "interval_seconds": 300,
                    "thresholds": {
                        "response_time_ms": 500,
                        "error_rate_percent": 5.0,
                        "throughput_min": 10
                    },
                    "triggers": ["scheduled", "api_call"]
                },
                "anomaly_detector": {
                    "enabled": True,
                    "interval_seconds": 120,
                    "thresholds": {
                        "z_score": 3.0,
                        "deviation_percent": 15.0
                    },
                    "triggers": ["continuous", "alert"]
                }
            },
            "diagnostics": {
                "scan_depth": "medium",
                "timeout_seconds": 300,
                "components": ["kernel", "memory", "network", "execution", "api"]
            },
            "healing": {
                "auto_heal": True,
                "strategies": ["restart", "patch", "reroute", "disable"],
                "max_attempts": 3,
                "cooldown_seconds": 600
            },
            "memory": {
                "working_memory_size": 1024,
                "persistent_storage_path": "data/memory",
                "retention_days": 30,
                "encryption_enabled": True
            },
            "logging": {
                "level": "INFO",
                "file_path": "logs/agent.log",
                "rotation_size_mb": 100,
                "retention_count": 10
            },
            "trust_threshold": 0.75,
            "telemetry_enabled": True,
            "environment": "production"
        }
    
    def _override_from_env(self, config_data: Dict[str, Any]) -> None:
        """Override configuration with environment variables"""
        # Example: SELF_HEALING_WATCHERS_STRESS_LOOP_ENABLED=false
        for key, value in os.environ.items():
            if key.startswith('SELF_HEALING_'):
                parts = key[13:].lower().split('_')
                self._set_nested_dict_value(config_data, parts, value)
    
    def _set_nested_dict_value(self, d: Dict[str, Any], keys: List[str], value: str) -> None:
        """Set a value in a nested dictionary based on a list of keys"""
        if not keys:
            return
        
        for i in range(len(keys) - 1):
            k = keys[i]
            if k not in d:
                d[k] = {}
            d = d[k]
        
        # Convert value to appropriate type
        last_key = keys[-1]
        if value.lower() in ('true', 'false'):
            d[last_key] = value.lower() == 'true'
        elif value.isdigit():
            d[last_key] = int(value)
        elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
            d[last_key] = float(value)
        else:
            d[last_key] = value
    
    def _create_config_objects(self, config_data: Dict[str, Any]) -> Config:
        """Create configuration objects from dictionary data"""
        watchers = {}
        for name, watcher_data in config_data["watchers"].items():
            watchers[name] = WatcherConfig(
                enabled=watcher_data["enabled"],
                interval_seconds=watcher_data["interval_seconds"],
                thresholds=watcher_data["thresholds"],
                triggers=watcher_data["triggers"]
            )
        
        diagnostics = DiagnosticConfig(
            scan_depth=config_data["diagnostics"]["scan_depth"],
            timeout_seconds=config_data["diagnostics"]["timeout_seconds"],
            components=config_data["diagnostics"]["components"]
        )
        
        healing = HealingConfig(
            auto_heal=config_data["healing"]["auto_heal"],
            strategies=config_data["healing"]["strategies"],
            max_attempts=config_data["healing"]["max_attempts"],
            cooldown_seconds=config_data["healing"]["cooldown_seconds"]
        )
        
        memory = MemoryConfig(
            working_memory_size=config_data["memory"]["working_memory_size"],
            persistent_storage_path=config_data["memory"]["persistent_storage_path"],
            retention_days=config_data["memory"]["retention_days"],
            encryption_enabled=config_data["memory"]["encryption_enabled"]
        )
        
        logging = LoggingConfig(
            level=config_data["logging"]["level"],
            file_path=config_data["logging"]["file_path"],
            rotation_size_mb=config_data["logging"]["rotation_size_mb"],
            retention_count=config_data["logging"]["retention_count"]
        )
        
        return Config(
            watchers=watchers,
            diagnostics=diagnostics,
            healing=healing,
            memory=memory,
            logging=logging,
            trust_threshold=config_data["trust_threshold"],
            telemetry_enabled=config_data["telemetry_enabled"],
            environment=config_data["environment"]
        )
    
    @property
    def config(self) -> Config:
        """Get the current configuration"""
        return self._config
    
    def reload(self) -> None:
        """Reload configuration from source"""
        self._load_config()
import logging
import os
import json
import time
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, Optional, List, Union

from self_healing_agent.config.settings import ConfigManager

class UniversalLogger:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UniversalLogger, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the logger with configuration settings"""
        self.config = ConfigManager().config
        self.logger = logging.getLogger("self_healing_agent")
        
        # Set log level
        level = getattr(logging, self.config.logging.level.upper(), logging.INFO)
        self.logger.setLevel(level)
        
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(self.config.logging.file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Configure rotating file handler
        file_handler = RotatingFileHandler(
            self.config.logging.file_path,
            maxBytes=self.config.logging.rotation_size_mb * 1024 * 1024,
            backupCount=self.config.logging.retention_count
        )
        
        # Configure console handler
        console_handler = logging.StreamHandler()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] [%(name)s] [%(module)s:%(lineno)d] - %(message)s'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Structured logging setup
        self.structured_log_path = os.path.join(log_dir, "structured_logs.jsonl")
    
    def _log_structured(self, level: str, component: str, message: str, 
                        event_type: str, data: Optional[Dict[str, Any]] = None) -> str:
        """Log a structured entry to the JSONL file"""
        log_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        log_entry = {
            "id": log_id,
            "timestamp": timestamp,
            "level": level,
            "component": component,
            "message": message,
            "event_type": event_type,
            "data": data or {}
        }
        
        with open(self.structured_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        return log_id
    
    def debug(self, component: str, message: str, event_type: str = "debug", 
              data: Optional[Dict[str, Any]] = None) -> str:
        """Log a debug message"""
        self.logger.debug(f"[{component}] {message}")
        return self._log_structured("DEBUG", component, message, event_type, data)
    
    def info(self, component: str, message: str, event_type: str = "info", 
             data: Optional[Dict[str, Any]] = None) -> str:
        """Log an info message"""
        self.logger.info(f"[{component}] {message}")
        return self._log_structured("INFO", component, message, event_type, data)
    
    def warning(self, component: str, message: str, event_type: str = "warning", 
                data: Optional[Dict[str, Any]] = None) -> str:
        """Log a warning message"""
        self.logger.warning(f"[{component}] {message}")
        return self._log_structured("WARNING", component, message, event_type, data)
    
    def error(self, component: str, message: str, event_type: str = "error", 
              data: Optional[Dict[str, Any]] = None, exc_info: bool = False) -> str:
        """Log an error message"""
        self.logger.error(f"[{component}] {message}", exc_info=exc_info)
        if exc_info and data is None:
            data = {}
        return self._log_structured("ERROR", component, message, event_type, data)
    
    def critical(self, component: str, message: str, event_type: str = "critical", 
                data: Optional[Dict[str, Any]] = None, exc_info: bool = True) -> str:
        """Log a critical message"""
        self.logger.critical(f"[{component}] {message}", exc_info=exc_info)
        if exc_info and data is None:
            data = {}
        return self._log_structured("CRITICAL", component, message, event_type, data)
    
    def diagnostic(self, component: str, diagnostic_data: Dict[str, Any], 
                  message: str = "Diagnostic data collected") -> str:
        """Log diagnostic data"""
        return self._log_structured(
            "INFO", component, message, "diagnostic", diagnostic_data
        )
    
    def healing(self, component: str, healing_action: str, 
               result: bool, details: Dict[str, Any]) -> str:
        """Log healing action"""
        message = f"Healing action '{healing_action}' {'succeeded' if result else 'failed'}"
        return self._log_structured(
            "INFO" if result else "WARNING", 
            component, 
            message, 
            "healing", 
            {"action": healing_action, "result": result, **details}
        )
    
        def metric(self, component: str, metrics: Dict[str, Union[float, int, str]], 
              context: Optional[Dict[str, Any]] = None) -> str:
        """Log metrics"""
        return self._log_structured(
            "INFO", 
            component, 
            f"Metrics collected for {component}", 
            "metric", 
            {"metrics": metrics, "context": context or {}}
        )
    
    def get_logs(self, 
                level: Optional[str] = None, 
                component: Optional[str] = None,
                event_type: Optional[str] = None, 
                start_time: Optional[str] = None,
                end_time: Optional[str] = None,
                limit: int = 100) -> List[Dict[str, Any]]:
        """Query logs with filters"""
        results = []
        
        with open(self.structured_log_path, "r") as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    
                    # Apply filters
                    if level and log_entry["level"] != level.upper():
                        continue
                    if component and log_entry["component"] != component:
                        continue
                    if event_type and log_entry["event_type"] != event_type:
                        continue
                    if start_time and log_entry["timestamp"] < start_time:
                        continue
                    if end_time and log_entry["timestamp"] > end_time:
                        continue
                    
                    results.append(log_entry)
                    if len(results) >= limit:
                        break
                except json.JSONDecodeError:
                    continue
        
        return results
    
    def reload_config(self):
        """Reload logger configuration"""
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Re-initialize with fresh config
        self._initialize()
import os
import json
import time
import pickle
import threading
import shutil
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
import uuid

from self_healing_agent.config.settings import ConfigManager
from self_healing_agent.core.logger import UniversalLogger

class MemoryCore:
    """
    Lightweight memory core with real-time working memory, persistent recall,
    and save-and-resume capability.
    """
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MemoryCore, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize memory systems"""
        self.config = ConfigManager().config
        self.logger = UniversalLogger()
        
        # Working memory (in-memory cache)
        self._working_memory = {}
        self._working_memory_size = 0
        self._working_memory_max_size = self.config.memory.working_memory_size * 1024 * 1024  # Convert to bytes
        
        # Persistent storage
        self._storage_path = Path(self.config.memory.persistent_storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        
        # Session management
        self._active_sessions = {}
        self._session_path = self._storage_path / "sessions"
        self._session_path.mkdir(exist_ok=True)
        
        # Initialize memory cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._periodic_cleanup,
            daemon=True
        )
        self._cleanup_thread.start()
        
        self.logger.info("memory_core", "Memory core initialized", "initialization")
    
    def _periodic_cleanup(self):
        """Periodically clean up old memory entries"""
        while True:
            try:
                # Sleep first to avoid immediate cleanup on startup
                time.sleep(3600)  # Run every hour
                
                self.cleanup_old_memories()
                self._trim_working_memory()
                
            except Exception as e:
                self.logger.error(
                    "memory_core", 
                    f"Error during memory cleanup: {str(e)}", 
                    "maintenance_error", 
                    exc_info=True
                )
    
    def _trim_working_memory(self):
        """Trim working memory if it exceeds the maximum size"""
        with self._lock:
            if self._working_memory_size <= self._working_memory_max_size:
                return
            
            # Sort items by last access time
            items = sorted(
                [(k, v) for k, v in self._working_memory.items()],
                key=lambda x: x[1].get("_last_accessed", 0)
            )
            
            # Remove oldest items until we're under the limit
            removed = 0
            for key, value in items:
                if self._working_memory_size <= self._working_memory_max_size * 0.8:  # 80% threshold
                    break
                
                item_size = self._estimate_size(value)
                del self._working_memory[key]
                self._working_memory_size -= item_size
                removed += 1
            
            self.logger.info(
                "memory_core", 
                f"Trimmed working memory, removed {removed} items",
                "maintenance"
            )
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate the memory size of an object in bytes"""
        try:
            return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        except:
            return 1024  # Default estimate if we can't pickle
    
    def store(self, key: str, value: Any, ttl: Optional[int] = None, 
             namespace: str = "default", persist: bool = False) -> bool:
        """
        Store a value in memory
        
        Args:
            key: The key to store the value under
            value: The value to store
            ttl: Time to live in seconds (None for no expiration)
            namespace: Namespace to store the value in
            persist: Whether to persist the value to disk
            
        Returns:
            bool: Success status
        """
        with self._lock:
            full_key = f"{namespace}:{key}"
            
            # Prepare memory entry
            entry = {
                "value": value,
                "created_at": time.time(),
                "last_accessed": time.time(),
                "ttl": ttl,
                "namespace": namespace,
                "metadata": {
                    "persisted": persist
                }
            }
            
            # Calculate size
            entry_size = self._estimate_size(entry)
            
            # Check if we're updating an existing entry
            if full_key in self._working_memory:
                old_size = self._estimate_size(self._working_memory[full_key])
                self._working_memory_size -= old_size
            
            # Store in working memory
            self._working_memory[full_key] = entry
            self._working_memory_size += entry_size
            
            # Persist if requested
            if persist:
                try:
                    self._persist_to_disk(namespace, key, entry)
                except Exception as e:
                    self.logger.error(
                        "memory_core", 
                        f"Failed to persist memory {full_key}: {str(e)}", 
                        "storage_error", 
                        exc_info=True
                    )
                    return False
            
            # Trim if necessary
            if self._working_memory_size > self._working_memory_max_size:
                self._trim_working_memory()
                
            return True
    
    def retrieve(self, key: str, namespace: str = "default", 
                default: Any = None, load_if_missing: bool = True) -> Any:
        """
        Retrieve a value from memory
        
        Args:
            key: The key to retrieve
            namespace: Namespace to retrieve from
            default: Default value if key not found
            load_if_missing: Whether to try loading from disk if not in memory
            
        Returns:
            The stored value or default
        """
        with self._lock:
            full_key = f"{namespace}:{key}"
            
            # Check working memory first
            if full_key in self._working_memory:
                entry = self._working_memory[full_key]
                
                # Check if expired
                if entry["ttl"] is not None:
                    if time.time() > entry["created_at"] + entry["ttl"]:
                        del self._working_memory[full_key]
                        self._working_memory_size -= self._estimate_size(entry)
                        
                        # Try to load from disk if it was persisted
                        if load_if_missing and entry["metadata"]["persisted"]:
                            disk_entry = self._load_from_disk(namespace, key)
                            if disk_entry:
                                return disk_entry["value"]
                        return default
                
                # Update last accessed time
                entry["last_accessed"] = time.time()
                return entry["value"]
            
            # Not in working memory, try to load from disk
            if load_if_missing:
                disk_entry = self._load_from_disk(namespace, key)
                if disk_entry:
                    # Add to working memory
                    self.store(
                        key, 
                        disk_entry["value"], 
                        ttl=disk_entry.get("ttl"),
                        namespace=namespace,
                        persist=False  # Already persisted
                    )
                    return disk_entry["value"]
            
            return default
    
    def _persist_to_disk(self, namespace: str, key: str, entry: Dict[str, Any]) -> bool:
        """Persist a memory entry to disk"""
        namespace_dir = self._storage_path / "persistent" / namespace
        namespace_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a safe filename from the key
        safe_key = hashlib.md5(key.encode()).hexdigest()
        file_path = namespace_dir / f"{safe_key}.json"
        
        # Prepare data for serialization
        try:
            serializable_entry = {
                "key": key,
                "value": entry["value"],
                "created_at": entry["created_at"],
                "last_accessed": entry["last_accessed"],
                "ttl": entry["ttl"],
                "namespace": namespace,
                "metadata": entry["metadata"]
            }
            
            # Write to temporary file first, then rename for atomicity
            temp_path = file_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(serializable_entry, f)
            
            # Atomic rename
            temp_path.rename(file_path)
            return True
            
        except (TypeError, ValueError) as e:
            # If JSON serialization fails, try pickle
            try:
                pickle_path = namespace_dir / f"{safe_key}.pickle"
                temp_path = pickle_path.with_suffix('.tmp')
                
                with open(temp_path, 'wb') as f:
                    pickle.dump(entry, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                temp_path.rename(pickle_path)
                return True
                
            except Exception as inner_e:
                self.logger.error(
                    "memory_core", 
                    f"Failed to persist memory {namespace}:{key}: {str(inner_e)}", 
                    "storage_error", 
                    exc_info=True
                )
                return False
        except Exception as e:
            self.logger.error(
                "memory_core", 
                f"Failed to persist memory {namespace}:{key}: {str(e)}", 
                "storage_error", 
                exc_info=True
            )
            return False
    
    def _load_from_disk(self, namespace: str, key: str) -> Optional[Dict[str, Any]]:
        """Load a memory entry from disk"""
        namespace_dir = self._storage_path / "persistent" / namespace
        if not namespace_dir.exists():
            return None
        
        # Create a safe filename from the key
        safe_key = hashlib.md5(key.encode()).hexdigest()
        json_path = namespace_dir / f"{safe_key}.json"
        pickle_path = namespace_dir / f"{safe_key}.pickle"
        
        # Try JSON first
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    entry = json.load(f)
                
                # Verify this is the correct key (hash collisions are possible)
                if entry.get("key") == key:
                    return entry
            except:
                pass
        
        # Try pickle as fallback
        if pickle_path.exists():
            try:
                with open(pickle_path, 'rb') as f:
                    entry = pickle.load(f)
                return entry
            except:
                pass
        
        return None
    
    def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete a value from memory and disk"""
        with self._lock:
            full_key = f"{namespace}:{key}"
            deleted = False
            
            # Remove from working memory
            if full_key in self._working_memory:
                entry_size = self._estimate_size(self._working_memory[full_key])
                del self._working_memory[full_key]
                self._working_memory_size -= entry_size
                deleted = True
            
            # Remove from disk
            namespace_dir = self._storage_path / "persistent" / namespace
            if namespace_dir.exists():
                safe_key = hashlib.md5(key.encode()).hexdigest()
                json_path = namespace_dir / f"{safe_key}.json"
                pickle_path = namespace_dir / f"{safe_key}.pickle"
                
                if json_path.exists():
                    json_path.unlink()
                    deleted = True
                
                if pickle_path.exists():
                    pickle_path.unlink()
                    deleted = True
            
            return deleted
    
    def cleanup_old_memories(self) -> int:
        """Clean up expired and old memories"""
        with self._lock:
            # Clean working memory
            current_time = time.time()
            expired_keys = []
            
            for full_key, entry in self._working_memory.items():
                if entry["ttl"] is not None and current_time > entry["created_at"] + entry["ttl"]:
                    expired_keys.append(full_key)
            
            for key in expired_keys:
                entry_size = self._estimate_size(self._working_memory[key])
                del self._working_memory[key]
                self._working_memory_size -= entry_size
            
            # Clean persistent storage
            retention_days = self.config.memory.retention_days
            cutoff_time = current_time - (retention_days * 86400)  # days to seconds
            
            removed_files = 0
            persistent_dir = self._storage_path / "persistent"
            
            if persistent_dir.exists():
                for namespace_dir in persistent_dir.iterdir():
                    if not namespace_dir.is_dir():
                        continue
                    
                    for file_path in namespace_dir.iterdir():
                        if not file_path.is_file():
                            continue
                        
                        try:
                            # Check file
                            # Check file modification time
                            file_mtime = file_path.stat().st_mtime
                            if file_mtime < cutoff_time:
                                file_path.unlink()
                                removed_files += 1
                        except Exception as e:
                            self.logger.warning(
                                "memory_core",
                                f"Error cleaning up memory file {file_path}: {str(e)}",
                                "cleanup_warning"
                            )

            self.logger.info(
                "memory_core",
                f"Cleaned up {len(expired_keys)} expired working memory items and {removed_files} old persisted files",
                "maintenance"
            )
            return len(expired_keys) + removed_files
