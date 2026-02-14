"""
Advanced Memory System for Threat Intelligence Agent.

Implements a multi-tier memory architecture:
1. Working Memory - Current investigation context (Redis)
2. Episodic Memory - Historical investigations (Vector DB)
3. Semantic Memory - Threat knowledge base (Vector DB)
4. Bead Memory - Attack chain correlation (Graph + Vector)

The Bead Memory system links related threat indicators across time,
creating "beads" on a "string" that represents an attack campaign.
"""

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import orjson
import structlog

logger = structlog.get_logger()


class MemoryType(Enum):
    """Types of memory in the system."""
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    BEAD = "bead"


class BeadType(Enum):
    """Types of beads in the Bead Memory system."""
    IOC = "ioc"
    TECHNIQUE = "technique"
    ACTOR = "actor"
    CAMPAIGN = "campaign"
    INCIDENT = "incident"
    ARTIFACT = "artifact"


@dataclass
class MemoryEntry:
    """Base memory entry."""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    ttl: Optional[int] = None  # Time to live in seconds
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "ttl": self.ttl,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        embedding = np.array(data["embedding"]) if data.get("embedding") else None
        return cls(
            id=data["id"],
            content=data["content"],
            embedding=embedding,
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            ttl=data.get("ttl"),
        )


@dataclass
class Bead:
    """
    A Bead represents a single piece of threat intelligence
    that can be linked to other beads in an attack chain.
    """
    id: str
    bead_type: BeadType
    value: str
    confidence: float
    first_seen: datetime
    last_seen: datetime
    embedding: Optional[np.ndarray] = None
    attributes: dict = field(default_factory=dict)
    source_ids: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bead_type": self.bead_type.value,
            "value": self.value,
            "confidence": self.confidence,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "attributes": self.attributes,
            "source_ids": self.source_ids,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Bead":
        embedding = np.array(data["embedding"]) if data.get("embedding") else None
        return cls(
            id=data["id"],
            bead_type=BeadType(data["bead_type"]),
            value=data["value"],
            confidence=data["confidence"],
            first_seen=datetime.fromisoformat(data["first_seen"]),
            last_seen=datetime.fromisoformat(data["last_seen"]),
            embedding=embedding,
            attributes=data.get("attributes", {}),
            source_ids=data.get("source_ids", []),
        )


@dataclass
class BeadString:
    """
    A BeadString represents a chain of related beads
    that form an attack campaign or incident timeline.
    """
    id: str
    name: str
    beads: list[str]  # Bead IDs in order
    links: list[tuple[str, str, float]]  # (bead_id, bead_id, strength)
    created_at: datetime
    updated_at: datetime
    campaign_id: Optional[str] = None
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "beads": self.beads,
            "links": self.links,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "campaign_id": self.campaign_id,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class MemoryBackend(ABC):
    """Abstract base class for memory backends."""
    
    @abstractmethod
    async def store(self, entry: MemoryEntry) -> bool:
        """Store a memory entry."""
        pass
    
    @abstractmethod
    async def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID."""
        pass
    
    @abstractmethod
    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[tuple[MemoryEntry, float]]:
        """Search for similar entries."""
        pass
    
    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all entries."""
        pass


class InMemoryBackend(MemoryBackend):
    """In-memory backend for development and testing."""
    
    def __init__(self):
        self.entries: dict[str, MemoryEntry] = {}
        self._lock = asyncio.Lock()
    
    async def store(self, entry: MemoryEntry) -> bool:
        async with self._lock:
            self.entries[entry.id] = entry
            return True
    
    async def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        return self.entries.get(entry_id)
    
    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[tuple[MemoryEntry, float]]:
        results = []
        
        for entry in self.entries.values():
            if entry.embedding is None:
                continue
            
            # Apply filters
            if filters:
                skip = False
                for key, value in filters.items():
                    if entry.metadata.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue
            
            # Cosine similarity
            similarity = np.dot(query_embedding, entry.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(entry.embedding)
            )
            results.append((entry, float(similarity)))
        
        results.sort(key=lambda x: -x[1])
        return results[:top_k]
    
    async def delete(self, entry_id: str) -> bool:
        async with self._lock:
            if entry_id in self.entries:
                del self.entries[entry_id]
                return True
            return False
    
    async def clear(self) -> bool:
        async with self._lock:
            self.entries.clear()
            return True


class RedisBackend(MemoryBackend):
    """Redis backend for working memory with TTL support."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        prefix: str = "ti_memory:",
    ):
        self.host = host
        self.port = port
        self.db = db
        self.prefix = prefix
        self._client = None
    
    async def _get_client(self):
        if self._client is None:
            import redis.asyncio as redis
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=False,
            )
        return self._client
    
    def _key(self, entry_id: str) -> str:
        return f"{self.prefix}{entry_id}"
    
    async def store(self, entry: MemoryEntry) -> bool:
        client = await self._get_client()
        data = orjson.dumps(entry.to_dict())
        
        if entry.ttl:
            await client.setex(self._key(entry.id), entry.ttl, data)
        else:
            await client.set(self._key(entry.id), data)
        
        return True
    
    async def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        client = await self._get_client()
        data = await client.get(self._key(entry_id))
        
        if data:
            return MemoryEntry.from_dict(orjson.loads(data))
        return None
    
    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[tuple[MemoryEntry, float]]:
        # Redis doesn't support vector search natively
        # For production, use Redis Stack with RediSearch
        # This is a simple scan-based implementation
        client = await self._get_client()
        results = []
        
        async for key in client.scan_iter(f"{self.prefix}*"):
            data = await client.get(key)
            if data:
                entry = MemoryEntry.from_dict(orjson.loads(data))
                if entry.embedding is not None:
                    similarity = np.dot(query_embedding, entry.embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(entry.embedding)
                    )
                    results.append((entry, float(similarity)))
        
        results.sort(key=lambda x: -x[1])
        return results[:top_k]
    
    async def delete(self, entry_id: str) -> bool:
        client = await self._get_client()
        result = await client.delete(self._key(entry_id))
        return result > 0
    
    async def clear(self) -> bool:
        client = await self._get_client()
        async for key in client.scan_iter(f"{self.prefix}*"):
            await client.delete(key)
        return True


class ChromaDBBackend(MemoryBackend):
    """ChromaDB backend for vector storage."""
    
    def __init__(
        self,
        collection_name: str = "threat_intel",
        persist_directory: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None
    
    def _get_collection(self):
        if self._collection is None:
            import chromadb
            
            if self.persist_directory:
                self._client = chromadb.PersistentClient(path=self.persist_directory)
            else:
                self._client = chromadb.Client()
            
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection
    
    async def store(self, entry: MemoryEntry) -> bool:
        collection = self._get_collection()
        
        embeddings = [entry.embedding.tolist()] if entry.embedding is not None else None
        
        collection.upsert(
            ids=[entry.id],
            embeddings=embeddings,
            documents=[entry.content],
            metadatas=[{
                **entry.metadata,
                "timestamp": entry.timestamp.isoformat(),
            }],
        )
        return True
    
    async def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        collection = self._get_collection()
        
        result = collection.get(ids=[entry_id], include=["documents", "embeddings", "metadatas"])
        
        if result["ids"]:
            embedding = np.array(result["embeddings"][0]) if result["embeddings"] else None
            metadata = result["metadatas"][0] if result["metadatas"] else {}
            timestamp = metadata.pop("timestamp", datetime.utcnow().isoformat())
            
            return MemoryEntry(
                id=entry_id,
                content=result["documents"][0],
                embedding=embedding,
                metadata=metadata,
                timestamp=datetime.fromisoformat(timestamp),
            )
        return None
    
    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[tuple[MemoryEntry, float]]:
        collection = self._get_collection()
        
        where = filters if filters else None
        
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where,
            include=["documents", "embeddings", "metadatas", "distances"],
        )
        
        entries = []
        for i, entry_id in enumerate(results["ids"][0]):
            embedding = np.array(results["embeddings"][0][i]) if results.get("embeddings") else None
            metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
            timestamp = metadata.pop("timestamp", datetime.utcnow().isoformat())
            distance = results["distances"][0][i] if results.get("distances") else 0.0
            
            entry = MemoryEntry(
                id=entry_id,
                content=results["documents"][0][i],
                embedding=embedding,
                metadata=metadata,
                timestamp=datetime.fromisoformat(timestamp),
            )
            
            # Convert distance to similarity (ChromaDB returns L2 distance)
            similarity = 1.0 / (1.0 + distance)
            entries.append((entry, similarity))
        
        return entries
    
    async def delete(self, entry_id: str) -> bool:
        collection = self._get_collection()
        collection.delete(ids=[entry_id])
        return True
    
    async def clear(self) -> bool:
        if self._client:
            self._client.delete_collection(self.collection_name)
            self._collection = None
        return True


class BeadMemory:
    """
    Bead Memory System for Attack Chain Correlation.
    
    This system links related threat indicators (beads) into
    chains (strings) that represent attack campaigns.
    
    Key concepts:
    - Bead: A single piece of threat intelligence (IOC, technique, etc.)
    - String: A chain of related beads forming an attack narrative
    - Link: A weighted connection between two beads
    """
    
    def __init__(
        self,
        vector_backend: MemoryBackend,
        similarity_threshold: float = 0.7,
        temporal_window: timedelta = timedelta(days=30),
    ):
        self.vector_backend = vector_backend
        self.similarity_threshold = similarity_threshold
        self.temporal_window = temporal_window
        
        self.beads: dict[str, Bead] = {}
        self.strings: dict[str, BeadString] = {}
        self.bead_to_strings: dict[str, set[str]] = {}  # bead_id -> string_ids
        
        self._lock = asyncio.Lock()
        self.logger = logger.bind(component="bead_memory")
    
    def _generate_bead_id(self, bead_type: BeadType, value: str) -> str:
        """Generate deterministic bead ID."""
        content = f"{bead_type.value}:{value}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def add_bead(
        self,
        bead_type: BeadType,
        value: str,
        embedding: np.ndarray,
        confidence: float = 1.0,
        attributes: Optional[dict] = None,
        source_id: Optional[str] = None,
    ) -> Bead:
        """
        Add a new bead or update existing one.
        
        If a bead with the same type and value exists, update its
        last_seen timestamp and merge attributes.
        """
        async with self._lock:
            bead_id = self._generate_bead_id(bead_type, value)
            now = datetime.utcnow()
            
            if bead_id in self.beads:
                # Update existing bead
                bead = self.beads[bead_id]
                bead.last_seen = now
                bead.confidence = max(bead.confidence, confidence)
                if attributes:
                    bead.attributes.update(attributes)
                if source_id and source_id not in bead.source_ids:
                    bead.source_ids.append(source_id)
            else:
                # Create new bead
                bead = Bead(
                    id=bead_id,
                    bead_type=bead_type,
                    value=value,
                    confidence=confidence,
                    first_seen=now,
                    last_seen=now,
                    embedding=embedding,
                    attributes=attributes or {},
                    source_ids=[source_id] if source_id else [],
                )
                self.beads[bead_id] = bead
                self.bead_to_strings[bead_id] = set()
            
            # Store in vector backend
            entry = MemoryEntry(
                id=bead_id,
                content=f"{bead_type.value}:{value}",
                embedding=embedding,
                metadata={
                    "bead_type": bead_type.value,
                    "value": value,
                    "confidence": confidence,
                },
            )
            await self.vector_backend.store(entry)
            
            self.logger.info(
                "bead_added",
                bead_id=bead_id,
                bead_type=bead_type.value,
                value=value[:50],
            )
            
            return bead
    
    async def find_related_beads(
        self,
        bead: Bead,
        top_k: int = 20,
    ) -> list[tuple[Bead, float]]:
        """Find beads related to the given bead."""
        if bead.embedding is None:
            return []
        
        # Search for similar beads
        results = await self.vector_backend.search(
            query_embedding=bead.embedding,
            top_k=top_k + 1,  # +1 to exclude self
        )
        
        related = []
        for entry, similarity in results:
            if entry.id == bead.id:
                continue
            
            if similarity < self.similarity_threshold:
                continue
            
            if entry.id in self.beads:
                related_bead = self.beads[entry.id]
                
                # Check temporal proximity
                time_diff = abs((bead.last_seen - related_bead.last_seen).total_seconds())
                if time_diff <= self.temporal_window.total_seconds():
                    related.append((related_bead, similarity))
        
        return related
    
    async def create_string(
        self,
        initial_beads: list[Bead],
        name: str,
        campaign_id: Optional[str] = None,
    ) -> BeadString:
        """Create a new bead string from initial beads."""
        async with self._lock:
            string_id = str(uuid4())[:8]
            now = datetime.utcnow()
            
            bead_ids = [b.id for b in initial_beads]
            
            # Calculate links based on similarity
            links = []
            for i, bead1 in enumerate(initial_beads):
                for bead2 in initial_beads[i+1:]:
                    if bead1.embedding is not None and bead2.embedding is not None:
                        similarity = np.dot(bead1.embedding, bead2.embedding) / (
                            np.linalg.norm(bead1.embedding) * np.linalg.norm(bead2.embedding)
                        )
                        if similarity >= self.similarity_threshold:
                            links.append((bead1.id, bead2.id, float(similarity)))
            
            # Calculate confidence
            confidence = np.mean([b.confidence for b in initial_beads])
            
            string = BeadString(
                id=string_id,
                name=name,
                beads=bead_ids,
                links=links,
                created_at=now,
                updated_at=now,
                campaign_id=campaign_id,
                confidence=confidence,
            )
            
            self.strings[string_id] = string
            
            # Update bead-to-string mapping
            for bead_id in bead_ids:
                self.bead_to_strings[bead_id].add(string_id)
            
            self.logger.info(
                "string_created",
                string_id=string_id,
                name=name,
                num_beads=len(bead_ids),
            )
            
            return string
    
    async def add_bead_to_string(
        self,
        string_id: str,
        bead: Bead,
    ) -> bool:
        """Add a bead to an existing string."""
        async with self._lock:
            if string_id not in self.strings:
                return False
            
            string = self.strings[string_id]
            
            if bead.id in string.beads:
                return True  # Already in string
            
            # Calculate links to existing beads
            new_links = []
            for existing_bead_id in string.beads:
                if existing_bead_id not in self.beads:
                    continue
                
                existing_bead = self.beads[existing_bead_id]
                if bead.embedding is not None and existing_bead.embedding is not None:
                    similarity = np.dot(bead.embedding, existing_bead.embedding) / (
                        np.linalg.norm(bead.embedding) * np.linalg.norm(existing_bead.embedding)
                    )
                    if similarity >= self.similarity_threshold:
                        new_links.append((bead.id, existing_bead_id, float(similarity)))
            
            # Add bead and links
            string.beads.append(bead.id)
            string.links.extend(new_links)
            string.updated_at = datetime.utcnow()
            
            # Recalculate confidence
            all_beads = [self.beads[bid] for bid in string.beads if bid in self.beads]
            string.confidence = np.mean([b.confidence for b in all_beads])
            
            self.bead_to_strings[bead.id].add(string_id)
            
            self.logger.info(
                "bead_added_to_string",
                string_id=string_id,
                bead_id=bead.id,
                new_links=len(new_links),
            )
            
            return True
    
    async def find_strings_for_bead(self, bead_id: str) -> list[BeadString]:
        """Find all strings containing a bead."""
        string_ids = self.bead_to_strings.get(bead_id, set())
        return [self.strings[sid] for sid in string_ids if sid in self.strings]
    
    async def merge_strings(
        self,
        string_ids: list[str],
        new_name: str,
    ) -> Optional[BeadString]:
        """Merge multiple strings into one."""
        async with self._lock:
            strings = [self.strings[sid] for sid in string_ids if sid in self.strings]
            
            if len(strings) < 2:
                return None
            
            # Collect all beads and links
            all_beads = set()
            all_links = []
            
            for s in strings:
                all_beads.update(s.beads)
                all_links.extend(s.links)
            
            # Remove duplicates from links
            unique_links = list(set(all_links))
            
            # Create merged string
            merged = BeadString(
                id=str(uuid4())[:8],
                name=new_name,
                beads=list(all_beads),
                links=unique_links,
                created_at=min(s.created_at for s in strings),
                updated_at=datetime.utcnow(),
                confidence=np.mean([s.confidence for s in strings]),
            )
            
            # Update mappings
            self.strings[merged.id] = merged
            
            for bead_id in all_beads:
                # Remove old string mappings
                for sid in string_ids:
                    self.bead_to_strings[bead_id].discard(sid)
                # Add new mapping
                self.bead_to_strings[bead_id].add(merged.id)
            
            # Remove old strings
            for sid in string_ids:
                del self.strings[sid]
            
            self.logger.info(
                "strings_merged",
                merged_id=merged.id,
                source_strings=string_ids,
                total_beads=len(all_beads),
            )
            
            return merged
    
    async def auto_correlate(
        self,
        bead: Bead,
        create_new_if_no_match: bool = True,
    ) -> Optional[BeadString]:
        """
        Automatically correlate a bead with existing strings.
        
        This is the main entry point for the Bead Memory system.
        It will:
        1. Find related beads
        2. Check if they belong to existing strings
        3. Add to existing string or create new one
        """
        # Find related beads
        related = await self.find_related_beads(bead)
        
        if not related:
            if create_new_if_no_match:
                # Create a new string with just this bead
                return await self.create_string(
                    initial_beads=[bead],
                    name=f"auto_{bead.bead_type.value}_{bead.id[:8]}",
                )
            return None
        
        # Find strings that related beads belong to
        candidate_strings: dict[str, float] = {}
        
        for related_bead, similarity in related:
            for string_id in self.bead_to_strings.get(related_bead.id, set()):
                if string_id in candidate_strings:
                    candidate_strings[string_id] = max(candidate_strings[string_id], similarity)
                else:
                    candidate_strings[string_id] = similarity
        
        if candidate_strings:
            # Add to the best matching string
            best_string_id = max(candidate_strings, key=candidate_strings.get)
            await self.add_bead_to_string(best_string_id, bead)
            return self.strings[best_string_id]
        
        if create_new_if_no_match:
            # Create new string with this bead and top related beads
            initial_beads = [bead] + [b for b, _ in related[:5]]
            return await self.create_string(
                initial_beads=initial_beads,
                name=f"auto_{bead.bead_type.value}_{bead.id[:8]}",
            )
        
        return None
    
    def get_string_graph(self, string_id: str) -> Optional[dict]:
        """Get a string as a graph representation."""
        if string_id not in self.strings:
            return None
        
        string = self.strings[string_id]
        
        nodes = []
        for bead_id in string.beads:
            if bead_id in self.beads:
                bead = self.beads[bead_id]
                nodes.append({
                    "id": bead_id,
                    "type": bead.bead_type.value,
                    "value": bead.value,
                    "confidence": bead.confidence,
                })
        
        edges = [
            {"source": link[0], "target": link[1], "weight": link[2]}
            for link in string.links
        ]
        
        return {
            "string_id": string_id,
            "name": string.name,
            "nodes": nodes,
            "edges": edges,
            "confidence": string.confidence,
        }
    
    def get_statistics(self) -> dict:
        """Get memory statistics."""
        return {
            "total_beads": len(self.beads),
            "total_strings": len(self.strings),
            "beads_by_type": {
                bt.value: sum(1 for b in self.beads.values() if b.bead_type == bt)
                for bt in BeadType
            },
            "avg_beads_per_string": np.mean([len(s.beads) for s in self.strings.values()])
            if self.strings else 0,
            "avg_links_per_string": np.mean([len(s.links) for s in self.strings.values()])
            if self.strings else 0,
        }


class ThreatIntelMemorySystem:
    """
    Complete memory system for the Threat Intelligence Agent.
    
    Coordinates all memory tiers:
    - Working Memory: Current investigation context
    - Episodic Memory: Historical investigations
    - Semantic Memory: Threat knowledge base
    - Bead Memory: Attack chain correlation
    """
    
    def __init__(
        self,
        working_memory: Optional[MemoryBackend] = None,
        episodic_memory: Optional[MemoryBackend] = None,
        semantic_memory: Optional[MemoryBackend] = None,
        bead_memory: Optional[BeadMemory] = None,
        working_memory_ttl: int = 3600,  # 1 hour
    ):
        self.working_memory = working_memory or InMemoryBackend()
        self.episodic_memory = episodic_memory or InMemoryBackend()
        self.semantic_memory = semantic_memory or InMemoryBackend()
        
        # Create bead memory with semantic memory as backend
        self.bead_memory = bead_memory or BeadMemory(
            vector_backend=self.semantic_memory,
        )
        
        self.working_memory_ttl = working_memory_ttl
        self.logger = logger.bind(component="memory_system")
    
    async def store_working_context(
        self,
        content: str,
        embedding: np.ndarray,
        context_type: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Store current investigation context."""
        entry_id = str(uuid4())[:12]
        
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            embedding=embedding,
            metadata={**(metadata or {}), "context_type": context_type},
            ttl=self.working_memory_ttl,
        )
        
        await self.working_memory.store(entry)
        
        self.logger.debug("stored_working_context", entry_id=entry_id, context_type=context_type)
        
        return entry_id
    
    async def store_investigation(
        self,
        investigation_id: str,
        summary: str,
        embedding: np.ndarray,
        findings: dict,
        related_iocs: list[str],
    ) -> bool:
        """Store completed investigation in episodic memory."""
        entry = MemoryEntry(
            id=investigation_id,
            content=summary,
            embedding=embedding,
            metadata={
                "findings": findings,
                "related_iocs": related_iocs,
                "investigation_type": "threat_intel",
            },
        )
        
        await self.episodic_memory.store(entry)
        
        self.logger.info("stored_investigation", investigation_id=investigation_id)
        
        return True
    
    async def store_threat_knowledge(
        self,
        knowledge_id: str,
        content: str,
        embedding: np.ndarray,
        knowledge_type: str,
        source: str,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Store threat knowledge in semantic memory."""
        entry = MemoryEntry(
            id=knowledge_id,
            content=content,
            embedding=embedding,
            metadata={
                **(metadata or {}),
                "knowledge_type": knowledge_type,
                "source": source,
            },
        )
        
        await self.semantic_memory.store(entry)
        
        self.logger.debug("stored_threat_knowledge", knowledge_id=knowledge_id, knowledge_type=knowledge_type)
        
        return True
    
    async def recall_relevant_context(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> dict[str, list[tuple[MemoryEntry, float]]]:
        """Recall relevant context from all memory tiers."""
        results = {
            "working": await self.working_memory.search(query_embedding, top_k),
            "episodic": await self.episodic_memory.search(query_embedding, top_k),
            "semantic": await self.semantic_memory.search(query_embedding, top_k),
        }
        
        return results
    
    async def process_ioc(
        self,
        ioc_type: str,
        ioc_value: str,
        embedding: np.ndarray,
        confidence: float = 1.0,
        source_id: Optional[str] = None,
    ) -> tuple[Bead, Optional[BeadString]]:
        """
        Process an IOC through the Bead Memory system.
        
        Returns the bead and any associated attack chain.
        """
        # Map IOC type to BeadType
        type_mapping = {
            "ipv4": BeadType.IOC,
            "ipv6": BeadType.IOC,
            "domain": BeadType.IOC,
            "url": BeadType.IOC,
            "md5": BeadType.IOC,
            "sha1": BeadType.IOC,
            "sha256": BeadType.IOC,
            "email": BeadType.IOC,
            "cve": BeadType.IOC,
            "mitre_technique": BeadType.TECHNIQUE,
        }
        
        bead_type = type_mapping.get(ioc_type, BeadType.IOC)
        
        # Add bead
        bead = await self.bead_memory.add_bead(
            bead_type=bead_type,
            value=ioc_value,
            embedding=embedding,
            confidence=confidence,
            attributes={"ioc_type": ioc_type},
            source_id=source_id,
        )
        
        # Auto-correlate with existing attack chains
        string = await self.bead_memory.auto_correlate(bead)
        
        return bead, string
    
    def get_memory_statistics(self) -> dict:
        """Get statistics from all memory tiers."""
        return {
            "bead_memory": self.bead_memory.get_statistics(),
            "timestamp": datetime.utcnow().isoformat(),
        }
