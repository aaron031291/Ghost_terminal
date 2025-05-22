#!/usr/bin/env python3
"""
Knowledge Management System for Grace Healing

This module provides a persistent knowledge store for error patterns, fixes, and learning
capabilities to improve self-healing over time. It maintains a database of known issues,
successful fixes, and provides interfaces for querying and updating this knowledge.
"""

import json
import os
import time
import logging
import sqlite3
import hashlib
import subprocess
import sys
import importlib
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import re
import threading

# Constants for dependency management
REQUIRED_LIBS = [
    "black",
    "pylint",
    "libcst",
    "parso",
    "autopep8",
    "pyflakes"
]
STATUS_FILE = os.path.join(os.getcwd(), "knowledge_env_status.json")

def ensure_dependencies():
    """
    Ensure all required dependencies are installed.
    Installs missing packages and generates a status report.
    """
    status_report = {
        "installed": [],
        "skipped": [],
        "failed": [],
        "timestamp": datetime.now().isoformat()
    }
    for lib in REQUIRED_LIBS:
        try:
            importlib.import_module(lib)
            status_report["skipped"].append(lib)
        except ImportError:
            print(f"[Knowledge Init] Installing missing package: {lib}")
            result = subprocess.call([sys.executable, "-m", "pip", "install", lib])
            if result == 0:
                status_report["installed"].append(lib)
            else:
                status_report["failed"].append(lib)
    
    # Save status report
    with open(STATUS_FILE, "w") as f:
        json.dump(status_report, f, indent=2)
    
    # Optional: print outcome
    print("\n[Knowledge Dependency Status]")
    print(json.dumps(status_report, indent=2))

# Mock imports for external dependencies
try:
    from cryptography.fernet import Fernet
except ImportError:
    # Mock implementation of Fernet for encryption
    class Fernet:
        def __init__(self, key):
            self.key = key
            
        def encrypt(self, data):
            if isinstance(data, str):
                data = data.encode()
            return b"MOCK_ENCRYPTED_" + data
            
        def decrypt(self, data):
            if data.startswith(b"MOCK_ENCRYPTED_"):
                return data[len(b"MOCK_ENCRYPTED_"):]
            return data

try:
    from flask import Flask, request, jsonify
except ImportError:
    # Mock Flask for API endpoints
    class Flask:
        def __init__(self, name):
            self.name = name
            self.routes = {}
            
        def route(self, path, methods=None):
            if methods is None:
                methods = ['GET']
                
            def decorator(f):
                self.routes[path] = (f, methods)
                return f
            return decorator
            
        def run(self, host='0.0.0.0', port=5000, debug=False):
            print(f"Mock Flask server running on {host}:{port}")
    
    class request:
        @staticmethod
        def get_json():
            return {}
    
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
logger = logging.getLogger("knowledge")

# Constants
DEFAULT_KNOWLEDGE_PATH = os.path.expanduser("~/.grace/knowledge_store.json")
DEFAULT_DB_PATH = os.path.expanduser("~/.grace/knowledge.db")
KNOWLEDGE_VERSION = "1.0.0"
MAX_CONFIDENCE = 1.0
MIN_CONFIDENCE = 0.0
TRUST_DECAY_FACTOR = 0.95  # 5% decay per day for unused fixes
DEFAULT_TTL_DAYS = 90  # Archive fixes older than this if unused


class KnowledgeStore:
    """
    Persistent storage for error patterns, fixes, and their metadata.
    Supports both file-based and SQLite storage backends.
    """
    
    def __init__(self, storage_path: str = None, use_sqlite: bool = True, 
                 encryption_key: str = None):
        """
        Initialize the knowledge store with the specified storage backend.
        
        Args:
            storage_path: Path to the knowledge store file or database
            use_sqlite: Whether to use SQLite (True) or JSON file storage (False)
            encryption_key: Optional encryption key for sensitive data
        """
        self.lock = threading.RLock()
        self.use_sqlite = use_sqlite
        self.encryption_key = encryption_key
        
        if encryption_key:
            self.fernet = Fernet(encryption_key.encode() if isinstance(encryption_key, str) 
                                else encryption_key)
        else:
            self.fernet = None
            
        if storage_path is None:
            storage_path = DEFAULT_DB_PATH if use_sqlite else DEFAULT_KNOWLEDGE_PATH
            
        self.storage_path = storage_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        
        if use_sqlite:
            self._init_sqlite_db()
        else:
            self._init_json_store()
            
        logger.info(f"Knowledge store initialized at {storage_path}")
    
    def _init_sqlite_db(self):
        """Initialize the SQLite database schema if it doesn't exist."""
        self.conn = sqlite3.connect(self.storage_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_entries (
            fingerprint TEXT PRIMARY KEY,
            error_type TEXT,
            error_message TEXT,
            fixes TEXT,  -- JSON array of fix objects
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            usage_count INTEGER DEFAULT 0,
            success_count INTEGER DEFAULT 0,
            trust_score REAL DEFAULT 0.5,
            tags TEXT,  -- JSON array of tags
            source TEXT,
            archived INTEGER DEFAULT 0
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_trail (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fingerprint TEXT,
            action TEXT,
            timestamp TIMESTAMP,
            details TEXT,
            FOREIGN KEY (fingerprint) REFERENCES knowledge_entries(fingerprint)
        )
        ''')
        
        # Initialize version if not present
        cursor.execute("SELECT value FROM metadata WHERE key = 'version'")
        version = cursor.fetchone()
        if not version:
            cursor.execute("INSERT INTO metadata VALUES (?, ?)", 
                          ('version', KNOWLEDGE_VERSION))
            cursor.execute("INSERT INTO metadata VALUES (?, ?)", 
                          ('created_at', datetime.now().isoformat()))
        
        self.conn.commit()
    
    def _init_json_store(self):
        """Initialize the JSON file storage if it doesn't exist."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    self.data = json.load(f)
                    
                # Ensure the version is compatible
                if 'version' not in self.data or self.data['version'] != KNOWLEDGE_VERSION:
                    logger.warning(f"Knowledge store version mismatch. Found: "
                                  f"{self.data.get('version')}, Expected: {KNOWLEDGE_VERSION}")
                    self._migrate_json_store()
            except json.JSONDecodeError:
                logger.error(f"Corrupted knowledge store at {self.storage_path}. Creating backup.")
                backup_path = f"{self.storage_path}.bak.{int(time.time())}"
                os.rename(self.storage_path, backup_path)
                self._create_empty_json_store()
        else:
            self._create_empty_json_store()
    
    def _create_empty_json_store(self):
        """Create a new empty JSON knowledge store."""
        self.data = {
            "version": KNOWLEDGE_VERSION,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "entries": {},
            "audit_trail": []
        }
        self._save_json_store()
    
    def _migrate_json_store(self):
        """Migrate an older version of the JSON store to the current version."""
        # For now, just create a backup and initialize a new store
        backup_path = f"{self.storage_path}.v{self.data.get('version', 'unknown')}"
        with open(backup_path, 'w') as f:
            json.dump(self.data, f, indent=2)
        
        logger.info(f"Created backup of old knowledge store at {backup_path}")
        self._create_empty_json_store()
    
    def _save_json_store(self):
        """Save the current state of the JSON store to disk."""
        with self.lock:
            self.data["updated_at"] = datetime.now().isoformat()
            with open(self.storage_path, 'w') as f:
                json.dump(self.data, f, indent=2)
    
    def _generate_fingerprint(self, error_message: str, error_type: str = None) -> str:
        """
        Generate a unique fingerprint for an error based on its message and type.
        
        Args:
            error_message: The error message text
            error_type: Optional error type/class
            
        Returns:
            A unique hash fingerprint for the error
        """
        # Normalize the error message to remove variable parts
        normalized_msg = self._normalize_error_message(error_message)
        
        # Create a hash of the normalized message and type
        hash_input = f"{normalized_msg}:{error_type or ''}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def _normalize_error_message(self, error_message: str) -> str:
        """
        Normalize an error message by removing variable parts like timestamps, PIDs, etc.
        
        Args:
            error_message: The raw error message
            
        Returns:
            Normalized error message with variable parts replaced by placeholders
        """
        if not error_message:
            return ""
            
        # Replace common variable patterns
        normalized = error_message
        
        # Replace timestamps
        normalized = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?',
                           '<TIMESTAMP>', normalized)
        
        # Replace file paths
        normalized = re.sub(r'(?:/[\w\-.]+)+(?:/[\w\-.]+)*', '<PATH>', normalized)
        
        # Replace hex addresses
        normalized = re.sub(r'0x[0-9a-fA-F]+', '<HEXADDR>', normalized)
        
        # Replace UUIDs
        normalized = re.sub(
            r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}',
            '<UUID>', normalized)
        
        # Replace numbers that are likely to be variable
        normalized = re.sub(r'\b\d+\b', '<NUM>', normalized)
        
        return normalized
    
    def add_entry(self, error_message: str, error_type: str = None, 
                 fixes: List[Dict] = None, source: str = "system") -> str:
        """
        Add a new knowledge entry for an error pattern.
        
        Args:
            error_message: The error message text
            error_type: Optional error type/class
            fixes: List of fix objects with solution, confidence, etc.
            source: Source of this knowledge (system, user, model, etc.)
            
        Returns:
            The fingerprint of the added entry
        """
        fingerprint = self._generate_fingerprint(error_message, error_type)
        
        with self.lock:
            if self.use_sqlite:
                return self._add_sqlite_entry(fingerprint, error_message, error_type, fixes, source)
            else:
                return self._add_json_entry(fingerprint, error_message, error_type, fixes, source)
    
    def _add_sqlite_entry(self, fingerprint: str, error_message: str, 
                         error_type: str, fixes: List[Dict], source: str) -> str:
        """Add an entry to the SQLite database."""
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        
        # Check if entry already exists
        cursor.execute("SELECT fingerprint FROM knowledge_entries WHERE fingerprint = ?", 
                      (fingerprint,))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing entry
            if fixes:
                cursor.execute("SELECT fixes FROM knowledge_entries WHERE fingerprint = ?", 
                              (fingerprint,))
                existing_fixes = json.loads(cursor.fetchone()[0] or "[]")
                
                # Merge fixes, avoiding duplicates
                existing_fix_texts = {fix.get('solution', '') for fix in existing_fixes}
                for fix in fixes:
                    if fix.get('solution', '') not in existing_fix_texts:
                        existing_fixes.append(fix)
                
                fixes_json = json.dumps(existing_fixes)
                
                cursor.execute("""
                UPDATE knowledge_entries 
                SET fixes = ?, updated_at = ?, source = ?
                WHERE fingerprint = ?
                """, (fixes_json, now, source, fingerprint))
            
            # Log the update in audit trail
            cursor.execute("""
            INSERT INTO audit_trail (fingerprint, action, timestamp, details)
            VALUES (?, ?, ?, ?)
            """, (fingerprint, "update", now, json.dumps({
                "source": source,
                "fixes_added": len(fixes) if fixes else 0
            })))
        else:
            # Create new entry
            fixes_json = json.dumps(fixes or [])
            tags_json = json.dumps(["new"])
            
            cursor.execute("""
            INSERT INTO knowledge_entries 
            (fingerprint, error_type, error_message, fixes, created_at, updated_at, 
             usage_count, success_count, trust_score, tags, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (fingerprint, error_type, error_message, fixes_json, now, now, 
                 0, 0, 0.5, tags_json, source))
            
            # Log the creation in audit trail
            cursor.execute("""
            INSERT INTO audit_trail (fingerprint, action, timestamp,
            details)
            VALUES (?, ?, ?, ?)
            """, (fingerprint, "create", now, json.dumps({
                "source": source,
                "error_type": error_type
            })))
        
        self.conn.commit()
        return fingerprint
    
    def _add_json_entry(self, fingerprint: str, error_message: str, 
                       error_type: str, fixes: List[Dict], source: str) -> str:
        """Add an entry to the JSON store."""
        now = datetime.now().isoformat()
        
        if fingerprint in self.data["entries"]:
            # Update existing entry
            if fixes:
                existing_fixes = self.data["entries"][fingerprint].get("fixes", [])
                
                # Merge fixes, avoiding duplicates
                existing_fix_texts = {fix.get('solution', '') for fix in existing_fixes}
                for fix in fixes:
                    if fix.get('solution', '') not in existing_fix_texts:
                        existing_fixes.append(fix)
                
                self.data["entries"][fingerprint]["fixes"] = existing_fixes
                self.data["entries"][fingerprint]["updated_at"] = now
                self.data["entries"][fingerprint]["source"] = source
            
            # Log the update in audit trail
            self.data["audit_trail"].append({
                "fingerprint": fingerprint,
                "action": "update",
                "timestamp": now,
                "details": {
                    "source": source,
                    "fixes_added": len(fixes) if fixes else 0
                }
            })
        else:
            # Create new entry
            self.data["entries"][fingerprint] = {
                "fingerprint": fingerprint,
                "error_type": error_type,
                "error_message": error_message,
                "fixes": fixes or [],
                "created_at": now,
                "updated_at": now,
                "usage_count": 0,
                "success_count": 0,
                "trust_score": 0.5,
                "tags": ["new"],
                "source": source,
                "archived": False
            }
            
            # Log the creation in audit trail
            self.data["audit_trail"].append({
                "fingerprint": fingerprint,
                "action": "create",
                "timestamp": now,
                "details": {
                    "source": source,
                    "error_type": error_type
                }
            })
        
        self._save_json_store()
        return fingerprint
    
    def get_entry(self, fingerprint: str) -> Optional[Dict]:
        """
        Retrieve a knowledge entry by its fingerprint.
        
        Args:
            fingerprint: The unique fingerprint of the entry
            
        Returns:
            The knowledge entry as a dictionary, or None if not found
        """
        with self.lock:
            if self.use_sqlite:
                cursor = self.conn.cursor()
                cursor.execute("""
                SELECT * FROM knowledge_entries 
                WHERE fingerprint = ? AND archived = 0
                """, (fingerprint,))
                row = cursor.fetchone()
                
                if row:
                    entry = dict(row)
                    entry["fixes"] = json.loads(entry["fixes"] or "[]")
                    entry["tags"] = json.loads(entry["tags"] or "[]")
                    return entry
                return None
            else:
                entry = self.data["entries"].get(fingerprint)
                if entry and not entry.get("archived", False):
                    return entry.copy()
                return None
    
    def find_entries(self, error_message: str = None, error_type: str = None, 
                    tags: List[str] = None, limit: int = 10) -> List[Dict]:
        """
        Find knowledge entries matching the given criteria.
        
        Args:
            error_message: Optional error message to match
            error_type: Optional error type to match
            tags: Optional list of tags to match
            limit: Maximum number of entries to return
            
        Returns:
            List of matching knowledge entries
        """
        with self.lock:
            if error_message:
                fingerprint = self._generate_fingerprint(error_message, error_type)
                exact_match = self.get_entry(fingerprint)
                if exact_match:
                    return [exact_match]
            
            # If no exact match or no error_message provided, search by criteria
            if self.use_sqlite:
                return self._find_sqlite_entries(error_message, error_type, tags, limit)
            else:
                return self._find_json_entries(error_message, error_type, tags, limit)
    
    def _find_sqlite_entries(self, error_message: str, error_type: str, 
                           tags: List[str], limit: int) -> List[Dict]:
        """Find entries in the SQLite database."""
        cursor = self.conn.cursor()
        query_parts = ["archived = 0"]
        params = []
        
        if error_type:
            query_parts.append("error_type = ?")
            params.append(error_type)
        
        if error_message:
            query_parts.append("error_message LIKE ?")
            params.append(f"%{error_message}%")
        
        query = f"""
        SELECT * FROM knowledge_entries 
        WHERE {' AND '.join(query_parts)}
        ORDER BY trust_score DESC, usage_count DESC
        LIMIT ?
        """
        params.append(limit)
        
        cursor.execute(query, params)
        results = []
        
        for row in cursor.fetchall():
            entry = dict(row)
            entry["fixes"] = json.loads(entry["fixes"] or "[]")
            entry["tags"] = json.loads(entry["tags"] or "[]")
            
            # Filter by tags if specified
            if tags and not all(tag in entry["tags"] for tag in tags):
                continue
                
            results.append(entry)
        
        return results
    
    def _find_json_entries(self, error_message: str, error_type: str, 
                         tags: List[str], limit: int) -> List[Dict]:
        """Find entries in the JSON store."""
        results = []
        
        for fingerprint, entry in self.data["entries"].items():
            if entry.get("archived", False):
                continue
                
            if error_type and entry.get("error_type") != error_type:
                continue
                
            if error_message and error_message.lower() not in entry.get("error_message", "").lower():
                continue
                
            if tags and not all(tag in entry.get("tags", []) for tag in tags):
                continue
                
            results.append(entry.copy())
            
            if len(results) >= limit:
                break
        
        # Sort by trust score and usage count
        results.sort(key=lambda x: (x.get("trust_score", 0), x.get("usage_count", 0)), reverse=True)
        return results
    
    def record_usage(self, fingerprint: str, success: bool = True) -> bool:
        """
        Record usage of a knowledge entry and update its trust score.
        
        Args:
            fingerprint: The unique fingerprint of the entry
            success: Whether the fix was successful
            
        Returns:
            True if the entry was found and updated, False otherwise
        """
        with self.lock:
            if self.use_sqlite:
                cursor = self.conn.cursor()
                cursor.execute("""
                SELECT usage_count, success_count, trust_score 
                FROM knowledge_entries 
                WHERE fingerprint = ?
                """, (fingerprint,))
                row = cursor.fetchone()
                
                if not row:
                    return False
                
                usage_count = row[0] + 1
                success_count = row[1] + (1 if success else 0)
                
                # Update trust score based on success rate
                trust_score = success_count / usage_count
                
                cursor.execute("""
                UPDATE knowledge_entries 
                SET usage_count = ?, success_count = ?, trust_score = ?, updated_at = ?
                WHERE fingerprint = ?
                """, (usage_count, success_count, trust_score, datetime.now().isoformat(), 
                     fingerprint))
                
                # Log the usage in audit trail
                cursor.execute("""
                INSERT INTO audit_trail (fingerprint, action, timestamp, details)
                VALUES (?, ?, ?, ?)
                """, (fingerprint, "usage", datetime.now().isoformat(), json.dumps({
                    "success": success,
                    "new_trust_score": trust_score
                })))
                
                self.conn.commit()
                return True
            else:
                entry = self.data["entries"].get(fingerprint)
                if not entry:
                    return False
                
                entry["usage_count"] = entry.get("usage_count", 0) + 1
                entry["success_count"] = entry.get("success_count", 0) + (1 if success else 0)
                
                # Update trust score based on success rate
                entry["trust_score"] = entry["success_count"] / entry["usage_count"]
                entry["updated_at"] = datetime.now().isoformat()
                
                # Log the usage in audit trail
                self.data["audit_trail"].append({
                    "fingerprint": fingerprint,
                    "action": "usage",
                    "timestamp": datetime.now().isoformat(),
                    "details": {
                        "success": success,
                        "new_trust_score": entry["trust_score"]
                    }
                })
                
                self._save_json_store()
                return True
    
    def add_fix(self, fingerprint: str, fix: Dict, source: str = "system") -> bool:
        """
        Add a new fix to an existing knowledge entry.
        
        Args:
            fingerprint: The unique fingerprint of the entry
            fix: The fix object with solution, confidence, etc.
            source: Source of this fix (system, user, model, etc.)
            
        Returns:
            True if the fix was added, False if the entry wasn't found
        """
        with self.lock:
            if self.use_sqlite:
                cursor = self.conn.cursor()
                cursor.execute("""
                SELECT fixes FROM knowledge_entries WHERE fingerprint = ?
                """, (fingerprint,))
                row = cursor.fetchone()
                
                if not row:
                    return False
                
                fixes = json.loads(row[0] or "[]")
                
                # Check if this fix already exists
                for existing_fix in fixes:
                    if existing_fix.get("solution") == fix.get("solution"):
                        # Update the existing fix
                        existing_fix.update(fix)
                        break
                else:
                    # Add the new fix
                    fix["added_at"] = datetime.now().isoformat()
                    fix["source"] = source
                    fixes.append(fix)
                
                cursor.execute("""
                UPDATE knowledge_entries 
                SET fixes = ?, updated_at = ?
                WHERE fingerprint = ?
                """, (json.dumps(fixes), datetime.now().isoformat(), fingerprint))
                
                # Log the addition in audit trail
                cursor.execute("""
                INSERT INTO audit_trail (fingerprint, action, timestamp, details)
                VALUES (?, ?, ?, ?)
                """, (fingerprint, "add_fix", datetime.now().isoformat(), json.dumps({
                    "source": source,
                    "fix": fix.get("solution", "")[:50]
                })))
                
                self.conn.commit()
                return True
            else:
                entry = self.data["entries"].get(fingerprint)
                if not entry:
                    return False
                
                fixes = entry.get("fixes", [])
                
                # Check if this fix already exists
                for existing_fix in fixes:
                    if existing_fix.get("solution") == fix.get("solution"):
                        # Update the existing fix
                        existing_fix.update(fix)
                        break
                else:
                    # Add the new fix
                    fix["added_at"] = datetime.now().isoformat()
                    fix["source"] = source
                    fixes.append(fix)
                
                entry["fixes"] = fixes
                entry["updated_at"] = datetime.now().isoformat()
                
                # Log the addition in audit trail
                self.data["audit_trail"].append({
                    "fingerprint": fingerprint,
                    "action": "add_fix",
                    "timestamp": datetime.now().isoformat(),
                    "details": {
                        "source": source,
                        "fix": fix.get("solution", "")[:50]
                    }
                })
                
                self._save_json_store()
                return True
    
    def archive_entry(self, fingerprint: str) -> bool:
        """
        Archive a knowledge entry so it's no longer returned in queries.
        
        Args:
            fingerprint: The unique fingerprint of the entry
            
        Returns:
            True if the entry was archived, False if not found
        """
        with self.lock:
            if self.use_sqlite:
                cursor = self.conn.cursor()
                cursor.execute("""
                UPDATE knowledge_entries 
                SET archived = 1, updated_at = ?
                WHERE fingerprint = ?
                """, (datetime.now().isoformat(), fingerprint))
                
                if cursor.rowcount == 0:
                    return False
                
                # Log the archival in audit trail
                cursor.execute("""
                INSERT INTO audit_trail (fingerprint, action, timestamp, details)
                VALUES (?, ?, ?, ?)
                """, (fingerprint, "archive", datetime.now().isoformat(), json.dumps({})))
                
                self.conn.commit()
                return True
            else:
                entry = self.data["entries"].get(fingerprint)
                if not entry:
                    return False
                
                entry["archived"] = True
                entry["updated_at"] = datetime.now().isoformat()
                
                # Log the archival in audit trail
                self.data["audit_trail"].append({
                    "fingerprint": fingerprint,
                    "action": "archive",
                    "timestamp": datetime.now().isoformat(),
                    "details": {}
                })
                
                self._save_json_store()
                return True
    
    def apply_trust_decay(self, days_threshold: int = 30, decay_factor: float = TRUST_DECAY_FACTOR):
        """
        Apply trust decay to entries that haven't been used recently.
        
        Args:
            days_threshold: Number of days of inactivity before applying decay
            decay_factor: Factor to multiply trust score by (0-1)
        """
        threshold_date = datetime.now().timestamp() - (days_threshold * 86400)
        
        with self.lock:
            if self.use_sqlite:
                cursor = self.conn.cursor()
                cursor.execute("""
                SELECT fingerprint, trust_score, updated_at 
                FROM knowledge_entries 
                WHERE archived = 0
                """)
                
                for row in cursor.fetchall():
                    fingerprint, trust_score, updated_at = row
                    updated_timestamp = datetime.fromisoformat(updated_at).timestamp()
                    
                    if updated_timestamp < threshold_date:
                        new_trust_score = max(MIN_CONFIDENCE, trust_score * decay_factor)
                        
                        cursor.execute("""
                        UPDATE knowledge_entries 
                        SET trust_score = ?
                        WHERE fingerprint = ?
                        """, (new_trust_score, fingerprint))
                
                self.conn.commit()
            else:
                for fingerprint, entry in self.data["entries"].items():
                    if entry.get("archived", False):
                        continue
                        
                    updated_at
                    updated_at = entry.get("updated_at")
                    if not updated_at:
                        continue
                        
                    updated_timestamp = datetime.fromisoformat(updated_at).timestamp()
                    
                    if updated_timestamp < threshold_date:
                        trust_score = entry.get("trust_score", 0.5)
                        entry["trust_score"] = max(MIN_CONFIDENCE, trust_score * decay_factor)
                
                self._save_json_store()
    
    def cleanup_old_entries(self, days_threshold: int = DEFAULT_TTL_DAYS):
        """
        Archive entries that haven't been used for a long time.
        
        Args:
            days_threshold: Number of days of inactivity before archiving
        """
        threshold_date = datetime.now().timestamp() - (days_threshold * 86400)
        
        with self.lock:
            if self.use_sqlite:
                cursor = self.conn.cursor()
                cursor.execute("""
                SELECT fingerprint, updated_at 
                FROM knowledge_entries 
                WHERE archived = 0
                """)
                
                for row in cursor.fetchall():
                    fingerprint, updated_at = row
                    updated_timestamp = datetime.fromisoformat(updated_at).timestamp()
                    
                    if updated_timestamp < threshold_date:
                        cursor.execute("""
                        UPDATE knowledge_entries 
                        SET archived = 1
                        WHERE fingerprint = ?
                        """, (fingerprint,))
                        
                        # Log the archival in audit trail
                        cursor.execute("""
                        INSERT INTO audit_trail (fingerprint, action, timestamp, details)
                        VALUES (?, ?, ?, ?)
                        """, (fingerprint, "auto_archive", datetime.now().isoformat(), json.dumps({
                            "reason": "ttl_expired",
                            "days_inactive": days_threshold
                        })))
                
                self.conn.commit()
            else:
                for fingerprint, entry in list(self.data["entries"].items()):
                    if entry.get("archived", False):
                        continue
                        
                    updated_at = entry.get("updated_at")
                    if not updated_at:
                        continue
                        
                    updated_timestamp = datetime.fromisoformat(updated_at).timestamp()
                    
                    if updated_timestamp < threshold_date:
                        entry["archived"] = True
                        
                        # Log the archival in audit trail
                        self.data["audit_trail"].append({
                            "fingerprint": fingerprint,
                            "action": "auto_archive",
                            "timestamp": datetime.now().isoformat(),
                            "details": {
                                "reason": "ttl_expired",
                                "days_inactive": days_threshold
                            }
                        })
                
                self._save_json_store()
    
    def close(self):
        """Close the knowledge store and release resources."""
        if self.use_sqlite and hasattr(self, 'conn'):
            self.conn.close()


class KnowledgeEngine:
    """
    Main interface for querying and updating the knowledge system.
    Provides high-level functions for error resolution and learning.
    """
    
    def __init__(self, storage_path: str = None, use_sqlite: bool = True, 
                 encryption_key: str = None):
        """
        Initialize the knowledge engine.
        
        Args:
            storage_path: Path to the knowledge store file or database
            use_sqlite: Whether to use SQLite (True) or JSON file storage (False)
            encryption_key: Optional encryption key for sensitive data
        """
        self.store = KnowledgeStore(storage_path, use_sqlite, encryption_key)
        self.similarity_threshold = 0.7  # Minimum similarity score to consider a match
        logger.info("Knowledge engine initialized")
    
    def query(self, error_message: str, error_type: str = None, 
             limit: int = 5) -> List[Dict]:
        """
        Query the knowledge base for fixes to an error.
        
        Args:
            error_message: The error message text
            error_type: Optional error type/class
            limit: Maximum number of entries to return
            
        Returns:
            List of knowledge entries with fixes, sorted by relevance
        """
        # First try exact fingerprint match
        fingerprint = self.store._generate_fingerprint(error_message, error_type)
        exact_match = self.store.get_entry(fingerprint)
        
        if exact_match and exact_match.get("fixes"):
            logger.info(f"Found exact match for error: {fingerprint}")
            return [exact_match]
        
        # If no exact match or no fixes, try similarity search
        logger.info(f"No exact match found, trying similarity search for: {error_message[:50]}...")
        similar_entries = self.store.find_entries(
            error_message=error_message,
            error_type=error_type,
            limit=limit
        )
        
        # Filter entries to only include those with fixes
        results = [entry for entry in similar_entries if entry.get("fixes")]
        
        if not results:
            logger.info("No fixes found in knowledge base")
            # Create a new entry for this error to track it
            self.store.add_entry(error_message, error_type)
        else:
            logger.info(f"Found {len(results)} potential fixes")
        
        return results
    
    def record_attempt(self, error_message: str, error_type: str, fix: str, 
                      success: bool) -> None:
        """
        Record an attempt to fix an error and its outcome.
        
        Args:
            error_message: The error message text
            error_type: The error type/class
            fix: The fix that was attempted
            success: Whether the fix was successful
        """
        fingerprint = self.store._generate_fingerprint(error_message, error_type)
        entry = self.store.get_entry(fingerprint)
        
        if not entry:
            # Create a new entry for this error
            fingerprint = self.store.add_entry(
                error_message, 
                error_type,
                fixes=[{
                    "solution": fix,
                    "confidence": 0.5,
                    "success_count": 1 if success else 0,
                    "attempt_count": 1
                }]
            )
        else:
            # Update existing entry
            self.store.record_usage(fingerprint, success)
            
            # Find or add the fix
            fix_exists = False
            for existing_fix in entry.get("fixes", []):
                if existing_fix.get("solution") == fix:
                    fix_exists = True
                    # Update fix stats in a separate call
                    existing_fix["attempt_count"] = existing_fix.get("attempt_count", 0) + 1
                    existing_fix["success_count"] = existing_fix.get("success_count", 0) + (
                        1 if success else 0)
                    existing_fix["confidence"] = existing_fix["success_count"] / existing_fix["attempt_count"]
                    self.store.add_fix(fingerprint, existing_fix)
                    break
            
            if not fix_exists:
                # Add this as a new fix
                self.store.add_fix(fingerprint, {
                    "solution": fix,
                    "confidence": 1.0 if success else 0.0,
                    "success_count": 1 if success else 0,
                    "attempt_count": 1
                })
        
        logger.info(f"Recorded fix attempt for {fingerprint}: success={success}")
    
    def elevate_fix_request(self, error_message: str, error_type: str, 
                           context: Dict = None) -> Dict:
        """
        Request a fix from an external ML model when no solution is found.
        
        Args:
            error_message: The error message text
            error_type: The error type/class
            context: Additional context about the error
            
        Returns:
            Response from the ML model with suggested fixes
        """
        # This would typically call an external ML service
        # For now, we'll just log the request and return a placeholder
        logger.info(f"Elevating fix request to ML model: {error_type}: {error_message[:50]}...")
        
        # In a real implementation, this would make an API call to an ML service
        # For demonstration, we'll just return a mock response
        response = {
            "success": True,
            "model_informed": True,
            "suggestions": [
                {
                    "solution": f"# Placeholder fix for {error_type}\n# This would be generated by an ML model",
                    "confidence": 0.7,
                    "reasoning": "This is a mock response. In a real system, this would contain the model's reasoning."
                }
            ]
        }
        
        # Store this suggestion in the knowledge base
        fingerprint = self.store._generate_fingerprint(error_message, error_type)
        for suggestion in response["suggestions"]:
            self.store.add_fix(fingerprint, suggestion, source="ml_model")
        
        return response
    
    def get_fix_suggestions(self, error_message: str, error_type: str = None, 
                          context: Dict = None, limit: int = 3) -> List[Dict]:
        """
        Get suggested fixes for an error, with confidence scores.
        
        Args:
            error_message: The error message text
            error_type: Optional error type/class
            context: Additional context about the error
            limit: Maximum number of suggestions to return
            
        Returns:
            List of suggested fixes with confidence scores
        """
        entries = self.query(error_message, error_type)
        
        if not entries:
            # No known fixes, try to elevate to ML model
            response = self.elevate_fix_request(error_message, error_type, context)
            suggestions = response.get("suggestions", [])
        else:
            # Collect all fixes from matching entries
            all_fixes = []
            for entry in entries:
                for fix in entry.get("fixes", []):
                    all_fixes.append({
                        "solution": fix.get("solution", ""),
                        "confidence": fix.get("confidence", 0.5),
                        "source": fix.get("source", "unknown"),
                        "entry_fingerprint": entry.get("fingerprint")
                    })
            
            # Sort by confidence and take top N
            suggestions = sorted(all_fixes, key=lambda x: x.get("confidence", 0), reverse=True)[:limit]
        
        return suggestions
    
    def perform_maintenance(self):
        """Perform routine maintenance tasks on the knowledge store."""
        logger.info("Starting knowledge base maintenance...")
        
        # Apply trust decay to old entries
        self.store.apply_trust_decay()
        
        # Archive very old entries
        self.store.cleanup_old_entries()
        
        logger.info("Knowledge base maintenance completed")
    
    def close(self):
        """Close the knowledge engine and release resources."""
        self.store.close()


class FusionMemory:
    """
    Short-term memory system that integrates with the knowledge engine
    for faster responses to recently seen errors.
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize the fusion memory.
        
        Args:
            max_size: Maximum number of entries to keep in memory
        """
        self.memory = {}
        self.max_size = max_size
        self.access_times = {}
        self.lock = threading.RLock()
    
    def store(self, key: str, value: Any) -> None:
        """
        Store a value in fusion memory.
        
        Args:
            key: The key to store the value under
            value: The value to store
        """
        with self.lock:
            # If at capacity, remove least recently used item
            if len(self.memory) >= self.max_size and key not in self.memory:
                oldest_key = min(self.access_times, key=self.access_times.get)
                del self.memory[oldest_key]
                del self.access_times[oldest_key]
            
            self.memory[key] = value
            self.access_times[key] = time.time()
    
    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from fusion memory.
        
        Args:
            key: The key to retrieve
            
        Returns:
            The stored value, or None if not found
        """
        with self.lock:
            if key in self.memory:
                self.access_times[key] = time.time()
                return self.memory[key]
            return None
    
    def clear(self) -> None:
        """Clear all entries from fusion memory."""
        with self.lock:
            self.memory.clear()
            self.access_times.clear()


class KnowledgeAPI:
    """
    API interface for the knowledge system, providing HTTP endpoints
    for querying and updating the knowledge base.
    """
    
    def __init__(self, engine: KnowledgeEngine, host: str = '0.0.0.0', port: int = 5000):
        """
        Initialize the knowledge API.
        
        Args:
            engine: The knowledge engine to use
            host: Host to bind the API server to
            port: Port to bind the API server to
        """
        self.engine = engine
        self.fusion_memory = FusionMemory()
        self.app = Flask("knowledge_api")
        self.host = host
        self.port = port
        
        self._setup_routes()
        logger.info(f"Knowledge API initialized on {host}:{port}")
    
    def _setup_routes(self):
        """Set up the API routes."""
        @self.app.route('/api/query', methods=['POST'])
        def query():
            data = request.get_json()
            error_message = data.get('error_message')
            error_type = data.get('error_type')
            
            if not error_message:
                return jsonify({"error": "error_message is required"}), 400
            
            # Check fusion memory first for faster response
            memory_key = f"query:{self.engine.store._generate_fingerprint(error_message, error_type)}"
            cached_result = self.fusion_memory.retrieve(memory_key)
            
            if cached_result:
                return jsonify({"results": cached_result, "source": "fusion_memory"})
            
            # Query the knowledge engine
            results = self.engine.query(error_message, error_type)
            
            # Store in fusion memory for future queries
            self.fusion_memory.store(memory_key, results)
            
            return jsonify({"results": results, "source": "knowledge_engine"})
        
        @self.app.route('/api/suggest', methods=['POST'])
        def suggest():
            data = request.get_json()
            error_message = data.get('error_message')
            error_type = data.get('error_type')
            context = data.get('context', {})
            
            if not error_message:
                return jsonify({"error": "error_message is required"}), 400
            
            suggestions = self.engine.get_fix_suggestions(error_message, error_type, context)
            return jsonify({"suggestions": suggestions})
        
        @self.app.route('/api/record', methods=['POST'])
        def record():
            data = request.get_json()
            error_message = data.get('error_message')
            error_type = data.get('error_type')
            fix = data.get('fix')
            success = data.get('success', False)
            
            if not all([error_message, fix]):
                return jsonify({"error": "error_message and fix are required"}), 400
            
            self.engine.record_attempt(error_message, error_type, fix, success)
            return jsonify({"status": "recorded"})
        
        @self.app.route('/api/maintenance', methods=['POST'])
        def maintenance():
            self.engine.perform_maintenance()
            return jsonify({"status": "maintenance completed"})
    
    def start(self, debug: bool = False):
        """
        Start the API server.
        
        Args:
            debug: Whether to run in debug mode
        """
        self.app.run(host=self.host, port=self.port, debug=debug)
    
    def stop(self):
        """Stop the API server and release resources."""
        # In a real implementation, this would properly shut down the Flask server
        self.engine.close()


def create_default_engine() -> KnowledgeEngine:
    """Create and return a default configured knowledge engine."""
    return KnowledgeEngine()


if __name__ == "__main__":
    # Ensure dependencies are installed
    ensure_dependencies()
    
    # Example usage
    engine = create_default_engine()
    api = KnowledgeAPI(engine)
    
    try:
        api.start(debug=True)
    except KeyboardInterrupt:
        logger.info("Shutting down Knowledge API...")
    finally:
        api.stop()
