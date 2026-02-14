"""
Unit tests for Threat Intelligence Agent Core.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from src.agent.core import (
    AgentState,
    TaskPriority,
    InvestigationType,
    ThreatContext,
    EnrichmentResult,
    ThreatIntelAgent,
)


class TestAgentState:
    """Test agent state transitions."""
    
    def test_initial_state_is_idle(self):
        """Agent should start in IDLE state."""
        agent = ThreatIntelAgent()
        assert agent.state == AgentState.IDLE
    
    def test_agent_id_generation(self):
        """Agent should generate unique ID if not provided."""
        agent1 = ThreatIntelAgent()
        agent2 = ThreatIntelAgent()
        assert agent1.agent_id != agent2.agent_id
        assert agent1.agent_id.startswith("ti_agent_")
    
    def test_custom_agent_id(self):
        """Agent should use provided ID."""
        agent = ThreatIntelAgent(agent_id="custom_agent")
        assert agent.agent_id == "custom_agent"


class TestThreatContext:
    """Test ThreatContext dataclass."""
    
    def test_threat_context_creation(self):
        """Test creating a threat context."""
        now = datetime.utcnow()
        context = ThreatContext(
            investigation_id="inv_123",
            investigation_type=InvestigationType.IOC_ENRICHMENT,
            priority=TaskPriority.HIGH,
            created_at=now,
            updated_at=now,
        )
        
        assert context.investigation_id == "inv_123"
        assert context.investigation_type == InvestigationType.IOC_ENRICHMENT
        assert context.priority == TaskPriority.HIGH
        assert context.status == "active"
        assert context.iocs == []
        assert context.findings == []
        assert context.severity_score == 0.0
    
    def test_threat_context_to_dict(self):
        """Test context serialization."""
        now = datetime.utcnow()
        context = ThreatContext(
            investigation_id="inv_456",
            investigation_type=InvestigationType.THREAT_HUNT,
            priority=TaskPriority.CRITICAL,
            created_at=now,
            updated_at=now,
            severity_score=0.85,
        )
        
        data = context.to_dict()
        
        assert data["investigation_id"] == "inv_456"
        assert data["investigation_type"] == "threat_hunt"
        assert data["priority"] == 0  # CRITICAL = 0
        assert data["severity_score"] == 0.85


class TestEnrichmentResult:
    """Test EnrichmentResult dataclass."""
    
    def test_enrichment_result_creation(self):
        """Test creating an enrichment result."""
        result = EnrichmentResult(
            ioc_type="ipv4",
            ioc_value="8.8.8.8",
            threat_score=0.75,
            confidence=0.9,
            classifications=[{"class": "c2", "confidence": 0.8}],
            related_iocs=[],
            threat_actors=[{"name": "APT29"}],
            campaigns=[],
            mitre_techniques=[{"id": "T1071", "name": "Application Layer Protocol"}],
            sources=["virustotal", "otx"],
        )
        
        assert result.ioc_type == "ipv4"
        assert result.ioc_value == "8.8.8.8"
        assert result.threat_score == 0.75
        assert len(result.threat_actors) == 1
        assert len(result.sources) == 2
    
    def test_enrichment_result_to_dict(self):
        """Test enrichment result serialization."""
        result = EnrichmentResult(
            ioc_type="domain",
            ioc_value="malware.com",
            threat_score=0.95,
            confidence=0.85,
            classifications=[],
            related_iocs=[],
            threat_actors=[],
            campaigns=[],
            mitre_techniques=[],
            first_seen=datetime(2024, 1, 1),
            last_seen=datetime(2024, 6, 1),
        )
        
        data = result.to_dict()
        
        assert data["ioc_type"] == "domain"
        assert data["threat_score"] == 0.95
        assert data["first_seen"] == "2024-01-01T00:00:00"


class TestThreatIntelAgent:
    """Test ThreatIntelAgent class."""
    
    @pytest.fixture
    def mock_memory_system(self):
        """Create a mock memory system."""
        memory = MagicMock()
        memory.process_ioc = AsyncMock(return_value=(MagicMock(), None))
        memory.recall_relevant_context = AsyncMock(return_value={})
        memory.semantic_memory = MagicMock()
        memory.semantic_memory.search = AsyncMock(return_value=[])
        memory.bead_memory = MagicMock()
        memory.bead_memory.strings = {}
        memory.get_memory_statistics = MagicMock(return_value={})
        return memory
    
    @pytest.fixture
    def mock_inference(self):
        """Create mock inference engine."""
        inference = MagicMock()
        inference.classify_threat = MagicMock(return_value={
            "predictions": [{"class_id": 0, "confidence": 0.8}]
        })
        inference.score_severity = MagicMock(return_value={
            "severity": "HIGH",
            "confidence": 0.85,
            "all_scores": {"HIGH": 0.85, "MEDIUM": 0.1, "LOW": 0.05, "CRITICAL": 0.0}
        })
        inference.map_to_mitre = MagicMock(return_value={
            "techniques": [{"id": "T1071", "confidence": 0.75}],
            "tactics": []
        })
        return inference
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, mock_memory_system):
        """Test agent initialization."""
        agent = ThreatIntelAgent(memory_system=mock_memory_system)
        
        with patch.object(agent, '_model', MagicMock()):
            with patch.object(agent, '_tokenizer', MagicMock()):
                with patch.object(agent, '_inference', MagicMock()):
                    agent._model.num_parameters = MagicMock(return_value=30000000)
                    await agent.initialize()
        
        assert agent.state == AgentState.READY
        assert agent._stats["start_time"] is not None
    
    @pytest.mark.asyncio
    async def test_start_investigation(self, mock_memory_system, mock_inference):
        """Test starting an investigation."""
        agent = ThreatIntelAgent(memory_system=mock_memory_system)
        agent.state = AgentState.READY
        agent._inference = mock_inference
        agent._get_embedding = MagicMock(return_value=np.random.rand(768))
        
        context = await agent.start_investigation(
            investigation_type=InvestigationType.IOC_ENRICHMENT,
            priority=TaskPriority.HIGH,
        )
        
        assert context.investigation_id.startswith("inv_")
        assert context.investigation_type == InvestigationType.IOC_ENRICHMENT
        assert context.priority == TaskPriority.HIGH
        assert agent._stats["total_investigations"] == 1
    
    @pytest.mark.asyncio
    async def test_list_investigations(self, mock_memory_system):
        """Test listing investigations."""
        agent = ThreatIntelAgent(memory_system=mock_memory_system)
        agent.state = AgentState.READY
        
        # Add some investigations
        now = datetime.utcnow()
        agent._investigations = {
            "inv_1": ThreatContext(
                investigation_id="inv_1",
                investigation_type=InvestigationType.IOC_ENRICHMENT,
                priority=TaskPriority.HIGH,
                created_at=now,
                updated_at=now,
            ),
            "inv_2": ThreatContext(
                investigation_id="inv_2",
                investigation_type=InvestigationType.THREAT_HUNT,
                priority=TaskPriority.MEDIUM,
                created_at=now,
                updated_at=now,
            ),
        }
        
        # List all
        all_investigations = await agent.list_investigations()
        assert len(all_investigations) == 2
        
        # Filter by type
        hunts = await agent.list_investigations(
            investigation_type=InvestigationType.THREAT_HUNT
        )
        assert len(hunts) == 1
        assert hunts[0].investigation_type == InvestigationType.THREAT_HUNT
    
    @pytest.mark.asyncio
    async def test_get_investigation(self, mock_memory_system):
        """Test getting a specific investigation."""
        agent = ThreatIntelAgent(memory_system=mock_memory_system)
        agent.state = AgentState.READY
        
        now = datetime.utcnow()
        agent._investigations["inv_test"] = ThreatContext(
            investigation_id="inv_test",
            investigation_type=InvestigationType.INCIDENT_ANALYSIS,
            priority=TaskPriority.CRITICAL,
            created_at=now,
            updated_at=now,
        )
        
        # Get existing
        context = await agent.get_investigation("inv_test")
        assert context is not None
        assert context.investigation_id == "inv_test"
        
        # Get non-existing
        context = await agent.get_investigation("inv_nonexistent")
        assert context is None
    
    def test_get_statistics(self, mock_memory_system):
        """Test getting agent statistics."""
        agent = ThreatIntelAgent(memory_system=mock_memory_system)
        agent.state = AgentState.READY
        agent._stats["start_time"] = datetime.utcnow() - timedelta(hours=1)
        agent._stats["total_enrichments"] = 100
        agent._stats["iocs_processed"] = 150
        
        stats = agent.get_statistics()
        
        assert stats["total_enrichments"] == 100
        assert stats["iocs_processed"] == 150
        assert stats["uptime"] is not None
    
    @pytest.mark.asyncio
    async def test_shutdown(self, mock_memory_system):
        """Test agent shutdown."""
        agent = ThreatIntelAgent(memory_system=mock_memory_system)
        agent.state = AgentState.READY
        
        await agent.shutdown()
        
        assert agent.state == AgentState.SHUTDOWN
    
    def test_callback_registration(self, mock_memory_system):
        """Test callback registration."""
        agent = ThreatIntelAgent(memory_system=mock_memory_system)
        
        finding_callback = MagicMock()
        alert_callback = MagicMock()
        
        agent.register_finding_callback(finding_callback)
        agent.register_alert_callback(alert_callback)
        
        assert agent._on_finding == finding_callback
        assert agent._on_alert == alert_callback


class TestThreatScoreCalculation:
    """Test threat score calculation logic."""
    
    def test_calculate_threat_score_critical(self):
        """Test threat score with critical severity."""
        agent = ThreatIntelAgent()
        
        severity_result = {
            "all_scores": {
                "CRITICAL": 0.9,
                "HIGH": 0.05,
                "MEDIUM": 0.03,
                "LOW": 0.02,
            }
        }
        external_data = {}
        
        score = agent._calculate_threat_score(severity_result, external_data)
        
        # Score should be close to 0.9 (CRITICAL weight = 1.0)
        assert score > 0.85
    
    def test_calculate_threat_score_with_external_boost(self):
        """Test threat score boosted by external data."""
        agent = ThreatIntelAgent()
        
        severity_result = {
            "all_scores": {
                "CRITICAL": 0.0,
                "HIGH": 0.5,
                "MEDIUM": 0.3,
                "LOW": 0.2,
            }
        }
        external_data = {
            "virustotal": {"malicious": True},
            "abuseipdb": {"suspicious": True},
        }
        
        score = agent._calculate_threat_score(severity_result, external_data)
        
        # Base score ~0.575, plus malicious (+0.2) and suspicious (+0.1) boosts
        assert score > 0.7


class TestAttackChainAssessment:
    """Test attack chain assessment logic."""
    
    def test_calculate_sophistication_high(self):
        """Test high sophistication calculation."""
        agent = ThreatIntelAgent()
        
        graph = {
            "nodes": [
                {"type": "technique"} for _ in range(6)
            ],
            "edges": [{"source": i, "target": i+1} for i in range(12)],
        }
        
        sophistication = agent._calculate_sophistication(graph)
        assert sophistication == "HIGH"
    
    def test_calculate_sophistication_low(self):
        """Test low sophistication calculation."""
        agent = ThreatIntelAgent()
        
        graph = {
            "nodes": [{"type": "ioc"}, {"type": "ioc"}],
            "edges": [{"source": 0, "target": 1}],
        }
        
        sophistication = agent._calculate_sophistication(graph)
        assert sophistication == "LOW"
    
    def test_calculate_risk_level_critical(self):
        """Test critical risk level calculation."""
        agent = ThreatIntelAgent()
        
        graph = {
            "confidence": 0.9,
            "nodes": [{"id": i} for i in range(15)],
        }
        
        risk = agent._calculate_risk_level(graph)
        assert risk == "CRITICAL"
    
    def test_determine_attack_stage(self):
        """Test attack stage determination."""
        agent = ThreatIntelAgent()
        
        graph = {
            "nodes": [
                {"type": "reconnaissance"},
                {"type": "initial_access"},
                {"type": "execution"},
                {"type": "lateral_movement"},
            ]
        }
        
        stage = agent._determine_attack_stage(graph)
        assert stage == "lateral_movement"
