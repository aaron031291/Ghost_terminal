Memory controller 

import json
import hashlib
import warnings
import uuid
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from threading import Lock
from typing import Dict, List, Optional, Any, Union

import numpy as np
import redis  # pip install redis
from google.cloud import firestore  # pip install google-cloud-firestore
from sqlalchemy import create_engine, Column, String, JSON, DateTime  # pip install sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Memory Tier Definitions
class MemoryTier(Enum):
    WORKING = auto()
    SHORT_TERM = auto()
    MEDIUM_TERM = auto()
    LONG_TERM = auto()

@dataclass
class MemoryItem:
    key: str
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    tier: MemoryTier = MemoryTier.WORKING

# Abstract Base Storage
class MemoryStorage(ABC):
    @abstractmethod
    def store(self, item: MemoryItem) -> bool:
        pass
        
    @abstractmethod
    def retrieve(self, key: str) -> Optional[MemoryItem]:
        pass
        
    @abstractmethod
    def purge_expired(self) -> int:
        pass

# Concrete Implementations
class WorkingMemory(MemoryStorage):
    """RAM-based ephemeral storage with process lifecycle"""
    def __init__(self):
        self._store: Dict[str, MemoryItem] = {}
        self.lock = Lock()
        
    def store(self, item: MemoryItem) -> bool:
        with self.lock:
            item.tier = MemoryTier.WORKING
            self._store[item.key] = item
        return True
        
    def retrieve(self, key: str) -> Optional[MemoryItem]:
        with self.lock:
            item = self._store.get(key)
            if item:
                item.last_accessed = datetime.now()
            return item
        
    def purge_expired(self) -> int:
        # Working memory clears only on process end
        return 0
        
    def clear_all(self):
        """Explicit flush for process restart"""
        with self.lock:
            count = len(self._store)
            self._store.clear()
            return count

class ShortTermMemory(MemoryStorage):
    """Redis-backed session storage with TTL"""
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=False
        )
        self.default_ttl = timedelta(hours=1)
        
    def store(self, item: MemoryItem) -> bool:
        item.tier = MemoryTier.SHORT_TERM
        serialized = json.dumps({
            "value": item.value,
            "metadata": item.metadata,
            "created_at": item.created_at.isoformat(),
            "last_accessed": item.last_accessed.isoformat()
        })
        success = self.redis.set(
            name=item.key,
            value=serialized,
            ex=int(self.default_ttl.total_seconds())
        )
        return bool(success)
        
    def retrieve(self, key: str) -> Optional[MemoryItem]:
        serialized = self.redis.get(key)
        if not serialized:
            return None
            
        # Refresh TTL on access
        self.redis.expire(key, int(self.default_ttl.total_seconds()))
            
        data = json.loads(serialized)
        return MemoryItem(
            key=key,
            value=data["value"],
            metadata=data["metadata"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            tier=MemoryTier.SHORT_TERM
        )
        
    def purge_expired(self) -> int:
        # Redis handles TTL automatically
        return 0

class MediumTermMemory(MemoryStorage):
    """Firestore-backed task-oriented storage"""
    def __init__(self, project_id="grace-ai"):
        self.db = firestore.Client(project=project_id)
        self.collection = "medium_term_memory"
        
    def store(self, item: MemoryItem) -> bool:
        item.tier = MemoryTier.MEDIUM_TERM
        doc_ref = self.db.collection(self.collection).document(item.key)
        doc_ref.set({
            "value": item.value,
            "metadata": item.metadata,
            "created_at": item.created_at,
            "last_accessed": item.last_accessed,
            "tier": item.tier.name
        })
        return True
        
    def retrieve(self, key: str) -> Optional[MemoryItem]:
        doc_ref = self.db.collection(self.collection).document(key)
        doc = doc_ref.get()
        if not doc.exists:
            return None
            
        data = doc.to_dict()
        return MemoryItem(
            key=key,
            value=data["value"],
            metadata=data["metadata"],
            created_at=data["created_at"],
            last_accessed=data["last_accessed"],
            tier=MemoryTier[data["tier"]]
        )
        
    def purge_expired(self) -> int:
        # Firestore would need a TTL policy set up
        # This is a placeholder for manual cleanup
        return 0

# SQLAlchemy models for long-term memory
Base = declarative_base()

class LongTermMemoryModel(Base):
    __tablename__ = "long_term_memory"
    key = Column(String(512), primary_key=True)
    value = Column(JSON)
    metadata = Column(JSON)
    embedding = Column(JSON)  # Stored as JSON array
    created_at = Column(DateTime)
    last_accessed = Column(DateTime)
    tier = Column(String(32))

class LongTermMemory(MemoryStorage):
    """PostgreSQL-backed persistent storage with vector support"""
    def __init__(self, connection_string="postgresql://user:pass@localhost/grace"):
        self.engine = create_engine(connection_string)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
    def store(self, item: MemoryItem) -> bool:
        item.tier = MemoryTier.LONG_TERM
            
        # Generate embedding if not provided
        if "embedding" not in item.metadata:
            item.metadata["embedding"] = self._generate_embedding(item.value)
            
        session = self.Session()
        try:
            model = LongTermMemoryModel(
                key=item.key,
                value=item.value,
                metadata=item.metadata,
                embedding=item.metadata["embedding"],
                created_at=item.created_at,
                last_accessed=item.last_accessed,
                tier=item.tier.name
            )
            session.merge(model)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
        
    def retrieve(self, key: str) -> Optional[MemoryItem]:
        session = self.Session()
        try:
            model = session.query(LongTermMemoryModel).filter_by(key=key).first()
            if not model:
                return None
                
            model.last_accessed = datetime.now()
            session.commit()
                
            return MemoryItem(
                key=model.key,
                value=model.value,
                metadata=model.metadata,
                created_at=model.created_at,
                last_accessed=model.last_accessed,
                tier=MemoryTier[model.tier]
            )
        finally:
            session.close()
        
    def semantic_search(self, query: str, k: int = 3) -> List[MemoryItem]:
        """Find similar items using vector similarity"""
        query_embedding = self._generate_embedding(query)
            
        # This would use pgvector in production
        # Simplified for example purposes
        session = self.Session()
        try:
            all_items = session.query(LongTermMemoryModel).all()
                
            # Calculate cosine similarity
            scored = []
            for item in all_items:
                if not item.embedding:
                    continue
                similarity = np.dot(query_embedding, item.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(item.embedding)
                )
                scored.append((similarity, item))
                
            # Return top k matches
            scored.sort(reverse=True, key=lambda x: x[0])
            return [
                MemoryItem(
                    key=item.key,
                    value=item.value,
                    metadata=item.metadata,
                    created_at=item.created_at,
                    last_accessed=datetime.now(),
                    tier=MemoryTier[item.tier]
                )
                for _, item in scored[:k]
            ]
        finally:
            session.close()
        
    def purge_expired(self) -> int:
        # Long-term memory persists indefinitely
        return 0
        
    def _generate_embedding(self, content: Any) -> List[float]:
        """Generate a simple embedding (replace with actual model in production)"""
        if isinstance(content, str):
            # Dummy embedding for illustration
            return [len(content)] * 10
        return [0.0] * 10

class MemoryController:
    """Orchestrates data flow across memory tiers"""
    def __init__(self):
        self.tiers = {
            MemoryTier.WORKING: WorkingMemory(),
            MemoryTier.SHORT_TERM: ShortTermMemory(),
            MemoryTier.MEDIUM_TERM: MediumTermMemory(),
            MemoryTier.LONG_TERM: LongTermMemory()
        }
        self.promotion_policies = {
            MemoryTier.WORKING: self._promote_from_working,
            MemoryTier.SHORT_TERM: self._promote_from_short_term,
            MemoryTier.MEDIUM_TERM: self._promote_from_medium_term
        }
        
    def store(self, key: str, value: Any, metadata: Dict = None, 
              tier: MemoryTier = MemoryTier.WORKING) -> bool:
        """Store data in the appropriate tier"""
        item = MemoryItem(
            key=key,
            value=value,
            metadata=metadata or {},
            tier=tier
        )
        return self.tiers[tier].store(item)
        
    def retrieve(self, key: str, search_lower_tiers: bool = True) -> Optional[MemoryItem]:
        """Retrieve data from memory, optionally cascading through tiers"""
        # Try each tier from highest to lowest
        for tier in [MemoryTier.LONG_TERM, MemoryTier.MEDIUM_TERM, 
                     MemoryTier.SHORT_TERM, MemoryTier.WORKING]:
            item = self.tiers[tier].retrieve(key)
            if item:
                # Consider promotion if accessed from lower tier
                if search_lower_tiers and tier != MemoryTier.LONG_TERM:
                    self._consider_promotion(item)
                return item
        return None
        
    def semantic_search(self, query: str, k: int = 3) -> List[MemoryItem]:
        """Search long-term memory using semantic similarity"""
        return self.tiers[MemoryTier.LONG_TERM].semantic_search(query, k)
        
    def promote_to_long_term(self, key: str) -> bool:
        """Explicitly promote an item to long-term memory"""
        # Find the item in any tier
        item = self.retrieve(key, search_lower_tiers=True)
        if not item:
            return False
            
        # Store in long-term memory
        return self.tiers[MemoryTier.LONG_TERM].store(item)
        
    def _consider_promotion(self, item: MemoryItem):
        """Apply promotion policies based on access patterns"""
        policy = self.promotion_policies.get(item.tier)
        if policy:
            policy(item)

    def _promote_from_working(self, item: MemoryItem):
        """Promote working memory items based on importance"""
        if item.metadata.get("importance", 0) > 0.8:
            self.tiers[MemoryTier.SHORT_TERM].store(item)

    def _promote_from_short_term(self, item: MemoryItem):
        """Promote short-term items based on session relevance"""
        if datetime.now() - item.created_at < timedelta(minutes=30):
            if item.metadata.get("access_count", 0) > 3:
                self.tiers[MemoryTier.MEDIUM_TERM].store(item)

    def _promote_from_medium_term(self, item: MemoryItem):
        """Promote medium-term items based on lasting value"""
        if item.metadata.get("retention_score", 0) > 0.7:
            self.tiers[MemoryTier.LONG_TERM].store(item)
        
    def run_maintenance(self):
        """Periodic cleanup and optimization"""
        total_purged = 0
        for tier in self.tiers.values():
            total_purged += tier.purge_expired()
        return total_purged

# Example Usage
if __name__ == "__main__":
    # Initialize memory system
    memory = MemoryController()
    
    # Store and retrieve data
    memory.store("user:123:prefs", {"theme": "dark", "font_size": 14},
                 tier=MemoryTier.SHORT_TERM)
    
    prefs = memory.retrieve("user:123:prefs")
    print(f"Retrieved preferences: {prefs.value if prefs else 'Not found'}")
    
    # Semantic search example
    memory.store("doc:ethics", "AI systems should respect human autonomy",
                 {"embedding": [0.1, 0.2, 0.3]},  # Real embeddings would come from a model
                tier=MemoryTier.LONG_TERM)
    
    results = memory.semantic_search("machine autonomy")
    print(f"Semantic search results: {[r.value for r in results]}")



