def __init__(self, prefix: str = "agent", redis_url: Optional[str] = None):
    """
    Initialize the lightning memory with Redis
    
    Args:
        prefix: Namespace prefix for Redis keys
        redis_url: Redis connection URL (defaults to env var or localhost)
    """
    self.prefix = prefix
    redis_url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    self.redis = redis.from_url(redis_url)
    
def _make_key(self, key: str) -> str:
    """Create a namespaced Redis key"""
    return f"{self.prefix}:{key}"
    
def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
    """
    Store a value in lightning memory
    
    Args:
        key: The key to store the value under
        value: The value to store (will be JSON serialized)
        ttl: Optional time-to-live in seconds
    """
    serialized = json.dumps(value)
    full_key = self._make_key(key)
    if ttl:
        self.redis.setex(full_key, ttl, serialized)
    else:
        self.redis.set(full_key, serialized)
        
def get(self, key: str) -> Any:
    """Retrieve a value from lightning memory"""
    full_key = self._make_key(key)
    value = self.redis.get(full_key)
    if value:
        return json.loads(value)
    return None
    
def delete(self, key: str) -> None:
    """Delete a value from lightning memory"""
    full_key = self._make_key(key)
    self.redis.delete(full_key)
    
def exists(self, key: str) -> bool:
    """Check if a key exists in lightning memory"""
    full_key = self._make_key(key)
    return bool(self.redis.exists(full_key))
    
def increment(self, key: str, amount: int = 1) -> int:
    """Atomically increment a counter"""
    full_key = self._make_key(key)
    return self.redis.incrby(full_key, amount)
    
def expire(self, key: str, seconds: int) -> None:
    """Set expiration on a key"""
    full_key = self._make_key(key)
    self.redis.expire(full_key, seconds)
    
def keys(self, pattern: str = "*") -> List[str]:
    """Get all keys matching pattern"""
    full_pattern = self._make_key(pattern)
    keys = self.redis.keys(full_pattern)
    # Strip prefix from returned keys
    prefix_len = len(self._make_key(""))
    return [k.decode('utf-8')[prefix_len:] for k in keys]

## 2. RAM Working Layer

Now, let's implement the volatile in-session memory:

```python:src/memory/working_memory.py```
from typing import Dict, Any, List, Optional
import time
import uuid

class WorkingMemory:
    """Volatile in-session memory for active processing"""
    
    def __init__(self):
        self._memory: Dict[str, Dict[str, Any]] = {}
        
    def store(self, data: Any, key: Optional[str] = None, 
              category: str = "general", metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store an item in working memory
        
        Args:
            data: The data to store
            key: Optional key (will generate UUID if not provided)
            category: Category for organizing memory
            metadata: Optional metadata to store with the item
            
        Returns:
            The key used to store the item
        """
        if key is None:
            key = str(uuid.uuid4())
            
        self._memory[key] = {
            "data": data,
            "category": category,
            "timestamp": time.time(),
            "metadata": metadata or {},
            "access_count": 0
        }
        return key
        
    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve an item from working memory
        
        Args:
            key: The key of the item to retrieve
            
        Returns:
            The stored data or None if not found
        """
        if key in self._memory:
            self._memory[key]["access_count"] += 1
            self._memory[key]["last_accessed"] = time.time()
            return self._memory[key]["data"]
        return None
        
    def get_with_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get the full memory entry including metadata"""
        if key in self._memory:
            self._memory[key]["access_count"] += 1
            self._memory[key]["last_accessed"] = time.time()
            return self._memory[key]
        return None
        
    def update(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update an existing memory item"""
        if key not in self._memory:
            return False
            
        self._memory[key]["data"] = data
        if metadata:
            self._memory[key]["metadata"].update(metadata)
        self._memory[key]["updated_at"] = time.time()
        return True
        
    def delete(self, key: str) -> bool:
        """Delete an item from working memory"""
        if key in self._memory:
            del self._memory[key]
            return True
        return False
        
    def list_by_category(self, category: str) -> List[str]:
        """List all keys in a specific category"""
        return [k for k, v in self._memory.items() if v["category"] == category]
        
    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Get all memory entries"""
        return self._memory
        
    def clear(self) -> None:
        """Clear all working memory"""
        self._memory = {}
        
    def get_recent(self, n: int = 5, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get the n most recent memory items, optionally filtered by category"""
        items = list(self._memory.values())
        if category:
            items = [item for item in items if item["category"] == category]
        
        return sorted(items, key=lambda x: x["timestamp"], reverse=True)[:n]
        
    def get_most_accessed(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the n most frequently accessed memory items"""
        items = list(self._memory.values())
        return sorted(items, key=lambda x: x.get("access_count", 0), reverse=True)[:n]
def __init__(self, project_id: str, storage_dir: str = "memory_storage"):
    """
    Initialize the librarian
    
    Args:
        project_id: Unique identifier for the project
        storage_dir: Directory for persistent storage
    """
    self.project_id = project_id
    self.lightning = LightningMemory(prefix=f"agent:{project_id}")
    self.working = WorkingMemory()
    self.storage = StorageMemory(os.path.join(storage_dir, project_id))
    
def remember(self, content: Any, tags: List[str] = None, 
             importance: int = 1, ttl: Optional[int] = None) -> str:
    """
    Store a memory across appropriate layers based on importance
    
    Args:
        content: The content to remember
        tags: List of tags for categorization and retrieval
        importance: Importance level (1-10)
        ttl: Optional time-to-live in seconds
        
    Returns:
        Memory ID
    """
    tags = tags or []
    timestamp = time.time()
    
    # Generate a unique ID for this memory
    content_hash = self._hash_content(content)
    memory_id = f"{int(timestamp)}_{content_hash[:8]}"
    
    memory_entry = {
        "id": memory_id,
        "content": content,
        "tags": tags,
        "importance": importance,
        "created_at": timestamp,
        "access_count": 0
    }
    
    # Store in working memory
    self.working.store(memory_entry, key=memory_id, 
                      category="memory", 
                      metadata={"tags": tags, "importance": importance})
    
    # High importance memories go to lightning memory
    if importance >= 7:
        self.lightning.set(f"memory:{memory_id}", memory_entry, ttl=ttl)
    
    # All memories go to persistent storage
    self.storage.save(memory_id, memory_entry)
    
    return memory_id

def recall(self, query: str = None, tags: List[str] = None, 
           limit: int = 5, min_importance: int = 0) -> List[Dict[str, Any]]:
    """
    Retrieve memories based on query and/or tags
    
    Args:
        query: Optional text query
        tags: Optional list of tags to filter by
        limit: Maximum number of results
        min_importance: Minimum importance level
        
    Returns:
        List of matching memories, ranked by relevance
    """
    # First check working memory
    working_memories = self._search_working_memory(tags, min_importance)
    
    # Then check persistent storage for additional matches
    storage_memories = self.storage.search(query, tags, limit * 2, min_importance)
    
    # Combine and deduplicate results
    all_memories = working_memories.copy()
    seen_ids = {m["id"] for m in all_memories}
    
    for memory in storage_memories:
        if memory["id"] not in seen_ids:
            all_memories.append(memory)
            seen_ids.add(memory["id"])
            
            # Also load into working memory for faster future access
            self.working.store(memory, key=memory["id"], 
                              category="memory", 
                              metadata={"tags": memory.get("tags", []), 
                                       "importance": memory.get("importance", 1)})
    
    # Rank results
    ranked_memories = self._rank_memories(all_memories, query, tags)
    
    # Update access counts for retrieved memories
    for memory in ranked_memories[:limit]:
        self._increment_access_count(memory["id"])
        
    return ranked_memories[:limit]

def forget(self, memory_id: str) -> bool:
    """
    Remove a memory from all storage layers
    
    Args:
        memory_id: ID of the memory to remove
        
    Returns:
        True if successful
    """
    success = True
    
    # Remove from working memory
    if not self.working.delete(memory_id):
        success = False
        
    # Remove from lightning memory
    self.lightning.delete(f"memory:{memory_id}")
    
    # Remove from storage
    if not self.storage.delete(memory_id):
        success = False
        
    return success

def update_importance(self, memory_id: str, new_importance: int) -> bool:
    """Update the importance level of a memory"""
    # Get the memory
    memory = self.storage.get(memory_id)
    if not memory:
        return False
        
    # Update importance
    memory["importance"] = new_importance
    
    # Update in all layers
    self.storage.save(memory_id, memory)
    
    working_entry = self.working.get_with_metadata(memory_id)
    if working_entry:
        working_entry["data"]["importance"] = new_importance
        working_entry["metadata"]["importance"] = new_importance
        self.working.update(memory_id, working_entry["data"], working_entry["metadata"])
    
    # For high importance, ensure it's in lightning memory
    if new_importance >= 7:
        self.lightning.set(f"memory:{memory_id}", memory)
    else:
        # Remove from lightning if importance dropped
        self.lightning.delete(f"memory:{memory_id}")
        
    return True

def get_recent_memories(self, limit: int = 10) -> List[Dict[str, Any]]:
    """Get the most recent memories"""
    return self.storage.get_recent(limit)

def get_important_memories(self, limit: int = 10) -> List[Dict[str, Any]]:
    """Get the most important memories"""
    return self.storage.get_by_importance(limit)

def _hash_content(self, content: Any) -> str:
    """Generate a hash for content"""
    if isinstance(content, dict) or isinstance(content, list):
        content = json.dumps(content, sort_keys=True)
    elif not isinstance(content, str):
        content = str(content)
        
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def _search_working_memory(self, tags: Optional[List[str]], min_importance: int) -> List[Dict[str, Any]]:
    """Search working memory for matching entries"""
    results = []
    
    for key in self.working.list_by_category("memory"):
        entry = self.working.get_with_metadata(key)
        if not entry:
            continue
            
        memory = entry["data"]
        
        # Check importance
        if memory.get("importance", 0) < min_importance:
            continue
            
        # Check tags
        if tags and not any(tag in memory.get("tags", []) for tag in tags):
            continue
            
        results.append(memory)
        
    return results

def _rank_memories(self, memories: List[Dict[str, Any]], 
                  query: Optional[str], tags: Optional[List[str]]) -> List[Dict[str, Any]]:
    """Rank memories by relevance to query and tags"""
    def score_memory(memory):
        score = 0
        
        # Importance factor
        score += memory.get("importance", 0) * 10
        
        # Recency factor (higher score for newer memories)
        age_hours = (time.time() - memory.get("created_at", 0)) / 3600
        recency_score = max(0, 100 - min(age_hours, 100))
        score += recency_score
        
        # Access count factor
        score += min(memory.get("access_count", 0) * 5, 50)
        
        # Tag matching
        if tags:
            memory_tags = memory.get("tags", [])
            matching_tags = sum(1 for tag in
