"""
Multi-tier Memory System for Threat Intelligence.

This module implements a sophisticated memory architecture:
- Working Memory: Current investigation context (Redis-backed)
- Episodic Memory: Historical investigations (Vector DB)
- Semantic Memory: Threat knowledge base (Vector DB)
- Bead Memory: Attack chain correlation (Graph + Vector)

The Bead Memory system is a novel approach that treats threat
intelligence as interconnected "beads" forming attack chains,
enabling automatic correlation across temporal and semantic dimensions.
"""

from .memory_system import (
    MemoryBackend,
    InMemoryBackend,
    RedisBackend,
    ChromaDBBackend,
    Bead,
    BeadString,
    BeadMemory,
    ThreatIntelMemorySystem,
    MemoryTier,
)

__all__ = [
    "MemoryBackend",
    "InMemoryBackend",
    "RedisBackend",
    "ChromaDBBackend",
    "Bead",
    "BeadString",
    "BeadMemory",
    "ThreatIntelMemorySystem",
    "MemoryTier",
]
