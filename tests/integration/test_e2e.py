"""
End-to-end integration tests for the Threat Intelligence Agent.

These tests verify the full workflow of the agent, including:
- Agent initialization and shutdown
- IOC enrichment through memory and providers
- Investigation lifecycle
- Attack chain correlation
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from src.agent.core import (
    AgentState,
    InvestigationType,
    TaskPriority,
    ThreatIntelAgent,
)
from src.memory.memory_system import (
    BeadType,
    InMemoryBackend,
    ThreatIntelMemorySystem,
)
from src.slm.model import TISLMConfig


@pytest.fixture
def memory_system():
    """Create a real memory system with in-memory backends."""
    return ThreatIntelMemorySystem(
        working_memory=InMemoryBackend(),
        episodic_memory=InMemoryBackend(),
        semantic_memory=InMemoryBackend(),
    )


@pytest.fixture
def mock_model():
    """Create a mock SLM model."""
    model = MagicMock()
    model.num_parameters = MagicMock(return_value=30000000)
    model.eval = MagicMock(return_value=model)
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.encode = MagicMock(return_value={
        "input_ids": [[101, 2000, 3000, 102]],
        "attention_mask": [[1, 1, 1, 1]],
    })
    tokenizer.extract_iocs = MagicMock(return_value=[
        {"type": "ipv4", "value": "192.168.1.1"}
    ])
    return tokenizer


@pytest.fixture
def mock_inference():
    """Create a mock inference engine."""
    inference = MagicMock()
    inference.classify_threat = MagicMock(return_value={
        "predictions": [
            {"class_id": 0, "class_name": "C2", "confidence": 0.85},
            {"class_id": 1, "class_name": "MALWARE", "confidence": 0.10},
        ]
    })
    inference.score_severity = MagicMock(return_value={
        "severity": "HIGH",
        "confidence": 0.9,
        "all_scores": {
            "CRITICAL": 0.05,
            "HIGH": 0.75,
            "MEDIUM": 0.15,
            "LOW": 0.05,
        }
    })
    inference.map_to_mitre = MagicMock(return_value={
        "techniques": [
            {"id": "T1071", "name": "Application Layer Protocol", "confidence": 0.8},
            {"id": "T1059", "name": "Command and Scripting Interpreter", "confidence": 0.6},
        ],
        "tactics": ["command-and-control", "execution"]
    })
    inference.get_embedding = MagicMock(return_value=MagicMock(
        cpu=MagicMock(return_value=MagicMock(
            numpy=MagicMock(return_value=np.random.rand(768))
        ))
    ))
    return inference


class TestAgentLifecycle:
    """Test agent lifecycle management."""
    
    @pytest.mark.asyncio
    async def test_full_agent_lifecycle(self, memory_system, mock_model, mock_tokenizer, mock_inference):
        """Test complete agent lifecycle: init -> ready -> processing -> shutdown."""
        agent = ThreatIntelAgent(
            agent_id="test_lifecycle_agent",
            memory_system=memory_system,
        )
        
        # Patch the model components
        with patch.object(agent, '_model', mock_model):
            with patch.object(agent, '_tokenizer', mock_tokenizer):
                with patch.object(agent, '_inference', mock_inference):
                    # Initialize
                    agent._model = mock_model
                    agent._tokenizer = mock_tokenizer
                    agent._inference = mock_inference
                    agent.state = AgentState.READY
                    agent._stats["start_time"] = datetime.utcnow()
                    
                    assert agent.state == AgentState.READY
                    
                    # Shutdown
                    await agent.shutdown()
                    assert agent.state == AgentState.SHUTDOWN


class TestIOCEnrichmentWorkflow:
    """Test IOC enrichment workflows."""
    
    @pytest.mark.asyncio
    async def test_enrich_ip_address(self, memory_system, mock_model, mock_tokenizer, mock_inference):
        """Test enriching an IP address IOC."""
        agent = ThreatIntelAgent(
            agent_id="test_enrich_agent",
            memory_system=memory_system,
        )
        
        # Setup agent
        agent._model = mock_model
        agent._tokenizer = mock_tokenizer
        agent._inference = mock_inference
        agent.state = AgentState.READY
        agent._stats["start_time"] = datetime.utcnow()
        
        # Override embedding method
        agent._get_embedding = MagicMock(return_value=np.random.rand(768))
        
        # Enrich an IOC
        result = await agent.enrich_ioc(
            ioc_type="ipv4",
            ioc_value="192.168.1.100"
        )
        
        # Verify result
        assert result.ioc_type == "ipv4"
        assert result.ioc_value == "192.168.1.100"
        assert result.threat_score >= 0
        assert result.confidence >= 0
        assert len(result.classifications) > 0
        assert len(result.mitre_techniques) > 0
        
        # Verify statistics updated
        assert agent._stats["iocs_processed"] == 1
        assert agent._stats["total_enrichments"] == 1
    
    @pytest.mark.asyncio
    async def test_enrich_multiple_iocs(self, memory_system, mock_model, mock_tokenizer, mock_inference):
        """Test enriching multiple IOCs."""
        agent = ThreatIntelAgent(
            agent_id="test_multi_enrich",
            memory_system=memory_system,
        )
        
        agent._model = mock_model
        agent._tokenizer = mock_tokenizer
        agent._inference = mock_inference
        agent.state = AgentState.READY
        agent._stats["start_time"] = datetime.utcnow()
        agent._get_embedding = MagicMock(return_value=np.random.rand(768))
        
        # Enrich multiple IOCs
        iocs = [
            ("ipv4", "10.0.0.1"),
            ("domain", "malware.example.com"),
            ("sha256", "a" * 64),
        ]
        
        results = []
        for ioc_type, ioc_value in iocs:
            result = await agent.enrich_ioc(ioc_type=ioc_type, ioc_value=ioc_value)
            results.append(result)
        
        assert len(results) == 3
        assert agent._stats["iocs_processed"] == 3


class TestInvestigationWorkflow:
    """Test investigation workflows."""
    
    @pytest.mark.asyncio
    async def test_create_and_track_investigation(self, memory_system, mock_model, mock_tokenizer, mock_inference):
        """Test creating and tracking an investigation."""
        agent = ThreatIntelAgent(
            agent_id="test_investigation",
            memory_system=memory_system,
        )
        
        agent._model = mock_model
        agent._tokenizer = mock_tokenizer
        agent._inference = mock_inference
        agent.state = AgentState.READY
        agent._stats["start_time"] = datetime.utcnow()
        agent._get_embedding = MagicMock(return_value=np.random.rand(768))
        
        # Start investigation
        context = await agent.start_investigation(
            investigation_type=InvestigationType.IOC_ENRICHMENT,
            priority=TaskPriority.HIGH,
            initial_iocs=[
                {"type": "ipv4", "value": "192.168.1.1"},
                {"type": "domain", "value": "c2.example.com"},
            ],
            metadata={"source": "siem_alert", "alert_id": "12345"}
        )
        
        # Verify investigation created
        assert context.investigation_id.startswith("inv_")
        assert context.investigation_type == InvestigationType.IOC_ENRICHMENT
        assert context.priority == TaskPriority.HIGH
        assert context.metadata["alert_id"] == "12345"
        
        # Get investigation
        retrieved = await agent.get_investigation(context.investigation_id)
        assert retrieved is not None
        assert retrieved.investigation_id == context.investigation_id
        
        # List investigations
        investigations = await agent.list_investigations()
        assert len(investigations) >= 1
    
    @pytest.mark.asyncio
    async def test_threat_hunt_workflow(self, memory_system, mock_model, mock_tokenizer, mock_inference):
        """Test threat hunting workflow."""
        agent = ThreatIntelAgent(
            agent_id="test_hunt",
            memory_system=memory_system,
        )
        
        agent._model = mock_model
        agent._tokenizer = mock_tokenizer
        agent._inference = mock_inference
        agent.state = AgentState.READY
        agent._stats["start_time"] = datetime.utcnow()
        agent._get_embedding = MagicMock(return_value=np.random.rand(768))
        
        # Start threat hunt
        context = await agent.hunt_threats(
            hypothesis="Possible lateral movement using PsExec",
            scope={"network": "internal"},
            time_range=timedelta(days=7),
        )
        
        # Verify hunt investigation created
        assert context.investigation_type == InvestigationType.THREAT_HUNT
        assert context.priority == TaskPriority.HIGH
        assert "hypothesis" in context.metadata
        assert len(context.mitre_mappings) > 0


class TestMemoryIntegration:
    """Test memory system integration."""
    
    @pytest.mark.asyncio
    async def test_ioc_stored_in_memory(self, memory_system, mock_model, mock_tokenizer, mock_inference):
        """Test that enriched IOCs are stored in memory."""
        agent = ThreatIntelAgent(
            agent_id="test_memory_storage",
            memory_system=memory_system,
        )
        
        agent._model = mock_model
        agent._tokenizer = mock_tokenizer
        agent._inference = mock_inference
        agent.state = AgentState.READY
        agent._stats["start_time"] = datetime.utcnow()
        agent._get_embedding = MagicMock(return_value=np.random.rand(768))
        
        # Enrich IOC
        await agent.enrich_ioc(
            ioc_type="ipv4",
            ioc_value="10.0.0.50"
        )
        
        # Verify bead created in memory
        bead_memory = agent.memory_system.bead_memory
        assert len(bead_memory.beads) > 0
        
        # Find our bead
        found = False
        for bead in bead_memory.beads.values():
            if bead.value == "10.0.0.50":
                found = True
                assert bead.bead_type == BeadType.IOC
                break
        
        assert found, "IOC bead not found in memory"
    
    @pytest.mark.asyncio
    async def test_memory_recall(self, memory_system, mock_model, mock_tokenizer, mock_inference):
        """Test recalling relevant context from memory."""
        agent = ThreatIntelAgent(
            agent_id="test_memory_recall",
            memory_system=memory_system,
        )
        
        agent._model = mock_model
        agent._tokenizer = mock_tokenizer
        agent._inference = mock_inference
        agent.state = AgentState.READY
        agent._stats["start_time"] = datetime.utcnow()
        
        # Create consistent embeddings for testing
        test_embedding = np.random.rand(768)
        agent._get_embedding = MagicMock(return_value=test_embedding)
        
        # Enrich some IOCs to populate memory
        for i in range(3):
            await agent.enrich_ioc(
                ioc_type="ipv4",
                ioc_value=f"192.168.1.{i}"
            )
        
        # Query memory
        results = await memory_system.recall_relevant_context(
            query_embedding=test_embedding,
            top_k=5,
        )
        
        # Memory should have entries
        stats = memory_system.get_memory_statistics()
        assert stats["bead_count"] >= 3


class TestAttackChainCorrelation:
    """Test attack chain correlation via Bead Memory."""
    
    @pytest.mark.asyncio
    async def test_related_iocs_form_chain(self, memory_system, mock_model, mock_tokenizer, mock_inference):
        """Test that related IOCs form attack chains."""
        agent = ThreatIntelAgent(
            agent_id="test_chain",
            memory_system=memory_system,
        )
        
        agent._model = mock_model
        agent._tokenizer = mock_tokenizer
        agent._inference = mock_inference
        agent.state = AgentState.READY
        agent._stats["start_time"] = datetime.utcnow()
        
        # Use similar embeddings to trigger correlation
        base_embedding = np.random.rand(768)
        call_count = [0]
        
        def get_similar_embedding(text):
            # Add small variation to simulate related but not identical
            call_count[0] += 1
            noise = np.random.rand(768) * 0.1  # Small noise
            return base_embedding + noise
        
        agent._get_embedding = get_similar_embedding
        
        # Enrich related IOCs in sequence
        investigation = await agent.start_investigation(
            investigation_type=InvestigationType.INCIDENT_ANALYSIS,
            priority=TaskPriority.CRITICAL,
        )
        
        # Simulate attack chain: C2 IP -> malware hash -> exfil domain
        await agent.enrich_ioc("ipv4", "10.20.30.40", investigation.investigation_id)
        await agent.enrich_ioc("sha256", "abc123" * 10 + "abcd", investigation.investigation_id)
        await agent.enrich_ioc("domain", "exfil.attacker.com", investigation.investigation_id)
        
        # Verify beads created
        bead_memory = agent.memory_system.bead_memory
        assert len(bead_memory.beads) >= 3


class TestCallbackSystem:
    """Test callback system for findings and alerts."""
    
    @pytest.mark.asyncio
    async def test_finding_callback_triggered(self, memory_system, mock_model, mock_tokenizer, mock_inference):
        """Test that finding callbacks are triggered."""
        agent = ThreatIntelAgent(
            agent_id="test_callbacks",
            memory_system=memory_system,
        )
        
        agent._model = mock_model
        agent._tokenizer = mock_tokenizer
        agent._inference = mock_inference
        agent.state = AgentState.READY
        agent._stats["start_time"] = datetime.utcnow()
        agent._get_embedding = MagicMock(return_value=np.random.rand(768))
        
        # Register callbacks
        findings = []
        alerts = []
        
        async def on_finding(finding):
            findings.append(finding)
        
        async def on_alert(alert):
            alerts.append(alert)
        
        agent.register_finding_callback(on_finding)
        agent.register_alert_callback(on_alert)
        
        # Start investigation and enrich IOC
        context = await agent.start_investigation(
            investigation_type=InvestigationType.IOC_ENRICHMENT,
        )
        
        await agent.enrich_ioc(
            ioc_type="ipv4",
            ioc_value="8.8.8.8",
            investigation_id=context.investigation_id,
        )
        
        # Verify finding callback was triggered
        assert len(findings) > 0
        assert findings[0]["type"] == "ioc_enrichment"


class TestStatistics:
    """Test agent statistics tracking."""
    
    @pytest.mark.asyncio
    async def test_statistics_tracking(self, memory_system, mock_model, mock_tokenizer, mock_inference):
        """Test that statistics are properly tracked."""
        agent = ThreatIntelAgent(
            agent_id="test_stats",
            memory_system=memory_system,
        )
        
        agent._model = mock_model
        agent._tokenizer = mock_tokenizer
        agent._inference = mock_inference
        agent.state = AgentState.READY
        agent._stats["start_time"] = datetime.utcnow()
        agent._get_embedding = MagicMock(return_value=np.random.rand(768))
        
        # Perform operations
        await agent.start_investigation(InvestigationType.IOC_ENRICHMENT)
        await agent.start_investigation(InvestigationType.THREAT_HUNT)
        
        await agent.enrich_ioc("ipv4", "1.1.1.1")
        await agent.enrich_ioc("domain", "test.com")
        await agent.enrich_ioc("sha256", "x" * 64)
        
        # Get statistics
        stats = agent.get_statistics()
        
        assert stats["total_investigations"] == 2
        assert stats["iocs_processed"] == 3
        assert stats["total_enrichments"] == 3
        assert stats["uptime"] is not None


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_enrichment_error_recovery(self, memory_system, mock_model, mock_tokenizer, mock_inference):
        """Test that agent recovers from enrichment errors."""
        agent = ThreatIntelAgent(
            agent_id="test_error_recovery",
            memory_system=memory_system,
        )
        
        agent._model = mock_model
        agent._tokenizer = mock_tokenizer
        agent._inference = mock_inference
        agent.state = AgentState.READY
        agent._stats["start_time"] = datetime.utcnow()
        agent._get_embedding = MagicMock(return_value=np.random.rand(768))
        
        # Make the inference raise an error for one call
        original_classify = mock_inference.classify_threat
        call_count = [0]
        
        def failing_classify(text):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Simulated failure")
            return original_classify(text)
        
        mock_inference.classify_threat = failing_classify
        
        # First call should succeed
        result1 = await agent.enrich_ioc("ipv4", "1.1.1.1")
        assert result1 is not None
        
        # Second call should fail
        with pytest.raises(Exception):
            await agent.enrich_ioc("ipv4", "2.2.2.2")
        
        # Agent should still be in READY state (recovered)
        assert agent.state == AgentState.READY
        
        # Third call should succeed
        result3 = await agent.enrich_ioc("ipv4", "3.3.3.3")
        assert result3 is not None
