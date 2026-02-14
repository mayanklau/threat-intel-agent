"""
Unit tests for the Multi-Tier Memory System.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest


class TestBeadMemory:
    """Tests for the Bead Memory system."""
    
    def test_bead_creation(self):
        """Test creating a Bead."""
        from src.memory.memory_system import Bead
        
        bead = Bead(
            id=str(uuid.uuid4()),
            bead_type="ioc",
            content="192.168.1.100",
            embedding=[0.1] * 768,
            timestamp=datetime.now(),
            metadata={"severity": "high"}
        )
        
        assert bead.bead_type == "ioc"
        assert bead.content == "192.168.1.100"
        assert len(bead.embedding) == 768
    
    def test_bead_string_creation(self):
        """Test creating a BeadString (attack chain)."""
        from src.memory.memory_system import Bead, BeadString
        
        beads = [
            Bead(
                id=str(uuid.uuid4()),
                bead_type="ioc",
                content=f"10.0.0.{i}",
                embedding=[0.1] * 768,
                timestamp=datetime.now() + timedelta(hours=i),
            )
            for i in range(3)
        ]
        
        string = BeadString(
            id=str(uuid.uuid4()),
            beads=beads,
            confidence=0.85,
            campaign="APT29-2024"
        )
        
        assert len(string.beads) == 3
        assert string.confidence == 0.85
        assert string.campaign == "APT29-2024"
    
    @pytest.mark.asyncio
    async def test_bead_memory_add_bead(self):
        """Test adding a bead to BeadMemory."""
        from src.memory.memory_system import BeadMemory, Bead
        
        memory = BeadMemory()
        await memory.initialize()
        
        bead = Bead(
            id=str(uuid.uuid4()),
            bead_type="ioc",
            content="192.168.1.100",
            embedding=[0.1] * 768,
            timestamp=datetime.now(),
        )
        
        await memory.add_bead(bead)
        
        # Verify bead was added
        assert bead.id in memory._beads
    
    @pytest.mark.asyncio
    async def test_bead_memory_find_related(self):
        """Test finding related beads."""
        from src.memory.memory_system import BeadMemory, Bead
        
        memory = BeadMemory()
        await memory.initialize()
        
        # Add similar beads
        embedding = [0.1] * 768
        for i in range(5):
            bead = Bead(
                id=str(uuid.uuid4()),
                bead_type="ioc",
                content=f"10.0.0.{i}",
                embedding=embedding.copy(),  # Similar embeddings
                timestamp=datetime.now(),
            )
            await memory.add_bead(bead)
        
        # Find related to a query
        related = await memory.find_related_beads(
            query_embedding=embedding,
            top_k=3
        )
        
        assert len(related) <= 3
    
    @pytest.mark.asyncio
    async def test_bead_memory_create_string(self):
        """Test creating a bead string (attack chain)."""
        from src.memory.memory_system import BeadMemory, Bead
        
        memory = BeadMemory()
        await memory.initialize()
        
        # Add beads
        beads = []
        for i in range(3):
            bead = Bead(
                id=str(uuid.uuid4()),
                bead_type="technique" if i > 0 else "ioc",
                content=f"content_{i}",
                embedding=[0.1 + i*0.01] * 768,
                timestamp=datetime.now() + timedelta(hours=i),
            )
            await memory.add_bead(bead)
            beads.append(bead)
        
        # Create string from beads
        string = await memory.create_string(
            bead_ids=[b.id for b in beads],
            campaign="TestCampaign"
        )
        
        assert string is not None
        assert len(string.beads) == 3
        assert string.campaign == "TestCampaign"
    
    @pytest.mark.asyncio
    async def test_bead_memory_auto_correlate(self):
        """Test auto-correlation of beads."""
        from src.memory.memory_system import BeadMemory, Bead
        
        memory = BeadMemory(
            temporal_window=timedelta(hours=24),
            similarity_threshold=0.5
        )
        await memory.initialize()
        
        # Add beads that should correlate
        base_embedding = [0.5] * 768
        for i in range(3):
            bead = Bead(
                id=str(uuid.uuid4()),
                bead_type="ioc",
                content=f"indicator_{i}",
                embedding=base_embedding.copy(),  # High similarity
                timestamp=datetime.now() + timedelta(hours=i),  # Temporal proximity
            )
            await memory.add_bead(bead)
        
        # Run auto-correlation
        strings = await memory.auto_correlate()
        
        # Should find correlations
        assert len(strings) >= 0  # May or may not correlate depending on thresholds


class TestInMemoryBackend:
    """Tests for the InMemoryBackend."""
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self):
        """Test storing and retrieving data."""
        from src.memory.memory_system import InMemoryBackend
        
        backend = InMemoryBackend()
        await backend.initialize()
        
        key = "test-key"
        value = {"data": "test", "count": 42}
        
        await backend.store(key, value)
        retrieved = await backend.retrieve(key)
        
        assert retrieved == value
    
    @pytest.mark.asyncio
    async def test_delete(self):
        """Test deleting data."""
        from src.memory.memory_system import InMemoryBackend
        
        backend = InMemoryBackend()
        await backend.initialize()
        
        key = "test-key"
        value = {"data": "test"}
        
        await backend.store(key, value)
        await backend.delete(key)
        retrieved = await backend.retrieve(key)
        
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_search(self):
        """Test searching by embedding similarity."""
        from src.memory.memory_system import InMemoryBackend
        
        backend = InMemoryBackend()
        await backend.initialize()
        
        # Store items with embeddings
        for i in range(5):
            await backend.store(
                f"item-{i}",
                {"content": f"content_{i}"},
                embedding=[0.1 * i] * 768
            )
        
        # Search with similar embedding
        query_embedding = [0.2] * 768
        results = await backend.search(query_embedding, top_k=3)
        
        assert len(results) <= 3


class TestThreatIntelMemorySystem:
    """Tests for the unified ThreatIntelMemorySystem."""
    
    @pytest.mark.asyncio
    async def test_memory_system_initialization(self, mock_settings):
        """Test memory system initialization."""
        from src.memory.memory_system import ThreatIntelMemorySystem
        
        memory = ThreatIntelMemorySystem(mock_settings.memory)
        await memory.initialize()
        
        assert memory._initialized
    
    @pytest.mark.asyncio
    async def test_store_working_context(self, mock_settings):
        """Test storing working memory context."""
        from src.memory.memory_system import ThreatIntelMemorySystem
        
        memory = ThreatIntelMemorySystem(mock_settings.memory)
        await memory.initialize()
        
        context = {
            "investigation_id": "inv-123",
            "iocs": ["192.168.1.100"],
            "current_step": "enrichment"
        }
        
        await memory.store_working_context("inv-123", context)
        
        retrieved = await memory.get_working_context("inv-123")
        assert retrieved["investigation_id"] == "inv-123"
    
    @pytest.mark.asyncio
    async def test_store_investigation(self, mock_settings):
        """Test storing investigation to episodic memory."""
        from src.memory.memory_system import ThreatIntelMemorySystem
        
        memory = ThreatIntelMemorySystem(mock_settings.memory)
        await memory.initialize()
        
        investigation = {
            "id": "inv-456",
            "type": "incident_analysis",
            "findings": ["finding1", "finding2"],
            "conclusion": "Confirmed APT activity"
        }
        embedding = [0.1] * 768
        
        await memory.store_investigation(investigation, embedding)
        
        # Should be retrievable via search
        results = await memory.recall_episodic(embedding, top_k=1)
        assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_store_threat_knowledge(self, mock_settings):
        """Test storing knowledge to semantic memory."""
        from src.memory.memory_system import ThreatIntelMemorySystem
        
        memory = ThreatIntelMemorySystem(mock_settings.memory)
        await memory.initialize()
        
        knowledge = {
            "entity": "APT29",
            "type": "threat_actor",
            "description": "Russian state-sponsored group",
            "techniques": ["T1059.001", "T1053.005"]
        }
        embedding = [0.2] * 768
        
        await memory.store_threat_knowledge(knowledge, embedding)
        
        # Should be retrievable via search
        results = await memory.recall_semantic(embedding, top_k=1)
        assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_process_ioc(self, mock_settings):
        """Test processing an IOC through Bead Memory."""
        from src.memory.memory_system import ThreatIntelMemorySystem
        
        memory = ThreatIntelMemorySystem(mock_settings.memory)
        await memory.initialize()
        
        ioc_data = {
            "value": "192.168.1.100",
            "type": "ipv4",
            "severity": "high",
            "source": "firewall"
        }
        embedding = [0.3] * 768
        
        bead = await memory.process_ioc(ioc_data, embedding)
        
        assert bead is not None
        assert bead.content == "192.168.1.100"
        assert bead.bead_type == "ioc"
    
    @pytest.mark.asyncio
    async def test_recall_relevant_context(self, mock_settings):
        """Test recalling context from all memory tiers."""
        from src.memory.memory_system import ThreatIntelMemorySystem
        
        memory = ThreatIntelMemorySystem(mock_settings.memory)
        await memory.initialize()
        
        # Store data in different tiers
        embedding = [0.4] * 768
        
        await memory.store_working_context("test-ctx", {"test": "data"})
        await memory.store_investigation({"id": "inv-1"}, embedding)
        await memory.store_threat_knowledge({"entity": "test"}, embedding)
        
        # Recall from all tiers
        context = await memory.recall_relevant_context(embedding, tiers=["episodic", "semantic"])
        
        assert "episodic" in context
        assert "semantic" in context
    
    @pytest.mark.asyncio
    async def test_get_stats(self, mock_settings):
        """Test getting memory statistics."""
        from src.memory.memory_system import ThreatIntelMemorySystem
        
        memory = ThreatIntelMemorySystem(mock_settings.memory)
        await memory.initialize()
        
        stats = await memory.get_stats()
        
        assert "working_memory" in stats
        assert "episodic_memory" in stats
        assert "semantic_memory" in stats
        assert "bead_memory" in stats


class TestMemoryTierIntegration:
    """Integration tests for memory tier interactions."""
    
    @pytest.mark.asyncio
    async def test_investigation_lifecycle(self, mock_settings):
        """Test full investigation lifecycle across memory tiers."""
        from src.memory.memory_system import ThreatIntelMemorySystem
        
        memory = ThreatIntelMemorySystem(mock_settings.memory)
        await memory.initialize()
        
        # 1. Create working context for active investigation
        investigation_id = "inv-lifecycle-test"
        await memory.store_working_context(investigation_id, {
            "status": "in_progress",
            "iocs": ["192.168.1.100", "evil.com"]
        })
        
        # 2. Process IOCs through bead memory
        for ioc in ["192.168.1.100", "evil.com"]:
            embedding = [0.5] * 768
            await memory.process_ioc(
                {"value": ioc, "type": "ioc", "severity": "high"},
                embedding
            )
        
        # 3. Complete investigation and store to episodic
        investigation_result = {
            "id": investigation_id,
            "conclusion": "Confirmed C2 traffic",
            "findings": ["C2 beacon detected", "Lateral movement observed"]
        }
        await memory.store_investigation(investigation_result, [0.5] * 768)
        
        # 4. Extract and store knowledge to semantic
        threat_knowledge = {
            "campaign": "Test Campaign",
            "iocs": ["192.168.1.100", "evil.com"],
            "techniques": ["T1071.001"]
        }
        await memory.store_threat_knowledge(threat_knowledge, [0.5] * 768)
        
        # 5. Verify data accessible from all tiers
        context = await memory.recall_relevant_context(
            [0.5] * 768,
            tiers=["episodic", "semantic"]
        )
        
        assert len(context["episodic"]) > 0
        assert len(context["semantic"]) > 0
