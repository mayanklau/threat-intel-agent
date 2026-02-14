"""
Unit tests for Threat Intelligence Agent REST API.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from fastapi.testclient import TestClient


# Mock the agent before importing the app
@pytest.fixture
def mock_agent():
    """Create a comprehensive mock agent."""
    agent = MagicMock()
    agent.state = MagicMock()
    agent.state.value = "ready"
    agent.agent_id = "test_agent"
    
    # Mock statistics
    agent.get_statistics = MagicMock(return_value={
        "agent_id": "test_agent",
        "state": "ready",
        "uptime": "1:00:00",
        "total_enrichments": 100,
        "total_investigations": 10,
        "iocs_processed": 150,
        "attacks_detected": 5,
        "active_investigations": 2,
        "memory_stats": {"working": 10, "episodic": 50, "semantic": 100},
    })
    
    # Mock memory system
    agent.memory_system = MagicMock()
    agent.memory_system.bead_memory = MagicMock()
    agent.memory_system.bead_memory.strings = {}
    agent.memory_system.bead_memory.get_string_graph = MagicMock(return_value=None)
    agent.memory_system.get_memory_statistics = MagicMock(return_value={})
    agent.memory_system.recall_relevant_context = AsyncMock(return_value={})
    
    # Mock inference
    agent._inference = MagicMock()
    agent._inference.map_to_mitre = MagicMock(return_value={
        "techniques": [{"id": "T1071", "name": "Application Layer Protocol", "confidence": 0.8}],
        "tactics": ["command-and-control"]
    })
    
    # Mock embedding function
    agent._get_embedding = MagicMock(return_value=np.random.rand(768))
    
    return agent


@pytest.fixture
def test_client(mock_agent):
    """Create test client with mocked agent."""
    with patch('src.api.main._agent', mock_agent):
        with patch('src.api.main.get_agent', return_value=mock_agent):
            from src.api.main import app
            client = TestClient(app)
            yield client


class TestHealthEndpoints:
    """Test health and metrics endpoints."""
    
    def test_health_check(self, test_client, mock_agent):
        """Test health check endpoint."""
        from src.agent.core import AgentState
        mock_agent.state = AgentState.READY
        
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
    
    def test_metrics_endpoint(self, test_client, mock_agent):
        """Test metrics endpoint."""
        response = test_client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_enrichments" in data
        assert "total_investigations" in data
        assert data["total_enrichments"] == 100
    
    def test_prometheus_metrics(self, test_client):
        """Test Prometheus metrics endpoint."""
        response = test_client.get("/metrics/prometheus")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"


class TestEnrichmentEndpoints:
    """Test IOC enrichment endpoints."""
    
    def test_enrich_single_ioc(self, test_client, mock_agent):
        """Test single IOC enrichment."""
        # Mock the enrichment result
        mock_result = MagicMock()
        mock_result.ioc_type = "ipv4"
        mock_result.ioc_value = "8.8.8.8"
        mock_result.threat_score = 0.75
        mock_result.confidence = 0.85
        mock_result.classifications = [{"class": "c2", "confidence": 0.8}]
        mock_result.related_iocs = []
        mock_result.threat_actors = [{"name": "APT29"}]
        mock_result.campaigns = []
        mock_result.mitre_techniques = [{"id": "T1071", "confidence": 0.7}]
        mock_result.sources = ["virustotal", "otx"]
        
        mock_agent.enrich_ioc = AsyncMock(return_value=mock_result)
        
        response = test_client.post(
            "/api/v1/enrich",
            json={
                "ioc_type": "ipv4",
                "ioc_value": "8.8.8.8"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["ioc_type"] == "ipv4"
        assert data["ioc_value"] == "8.8.8.8"
        assert data["threat_score"] == 0.75
    
    def test_enrich_with_invalid_type(self, test_client, mock_agent):
        """Test enrichment with valid request format."""
        mock_result = MagicMock()
        mock_result.ioc_type = "unknown"
        mock_result.ioc_value = "test"
        mock_result.threat_score = 0.0
        mock_result.confidence = 0.0
        mock_result.classifications = []
        mock_result.related_iocs = []
        mock_result.threat_actors = []
        mock_result.campaigns = []
        mock_result.mitre_techniques = []
        mock_result.sources = []
        
        mock_agent.enrich_ioc = AsyncMock(return_value=mock_result)
        
        response = test_client.post(
            "/api/v1/enrich",
            json={
                "ioc_type": "unknown",
                "ioc_value": "test"
            }
        )
        
        assert response.status_code == 200
    
    def test_batch_enrichment(self, test_client, mock_agent):
        """Test batch IOC enrichment."""
        mock_context = MagicMock()
        mock_context.investigation_id = "inv_batch_123"
        
        mock_agent.start_investigation = AsyncMock(return_value=mock_context)
        
        response = test_client.post(
            "/api/v1/enrich/batch",
            json={
                "iocs": [
                    {"ioc_type": "ipv4", "ioc_value": "8.8.8.8"},
                    {"ioc_type": "domain", "ioc_value": "example.com"},
                ],
                "create_investigation": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"
        assert data["investigation_id"] == "inv_batch_123"
        assert data["ioc_count"] == 2


class TestInvestigationEndpoints:
    """Test investigation management endpoints."""
    
    def test_create_investigation(self, test_client, mock_agent):
        """Test creating an investigation."""
        from src.agent.core import InvestigationType, TaskPriority
        
        now = datetime.utcnow()
        mock_context = MagicMock()
        mock_context.investigation_id = "inv_new_123"
        mock_context.investigation_type = InvestigationType.IOC_ENRICHMENT
        mock_context.priority = TaskPriority.HIGH
        mock_context.status = "active"
        mock_context.created_at = now
        mock_context.updated_at = now
        mock_context.iocs = []
        mock_context.findings = []
        mock_context.severity_score = 0.0
        mock_context.confidence_score = 0.0
        
        mock_agent.start_investigation = AsyncMock(return_value=mock_context)
        
        response = test_client.post(
            "/api/v1/investigations",
            json={
                "investigation_type": "ioc_enrichment",
                "priority": "HIGH",
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["investigation_id"] == "inv_new_123"
        assert data["investigation_type"] == "ioc_enrichment"
    
    def test_get_investigation(self, test_client, mock_agent):
        """Test getting an investigation."""
        from src.agent.core import InvestigationType, TaskPriority
        
        now = datetime.utcnow()
        mock_context = MagicMock()
        mock_context.investigation_id = "inv_123"
        mock_context.investigation_type = InvestigationType.THREAT_HUNT
        mock_context.priority = TaskPriority.CRITICAL
        mock_context.status = "completed"
        mock_context.created_at = now
        mock_context.updated_at = now
        mock_context.iocs = [{"type": "ipv4", "value": "1.2.3.4"}]
        mock_context.findings = [{"type": "threat", "severity": "high"}]
        mock_context.severity_score = 0.85
        mock_context.confidence_score = 0.9
        
        mock_agent.get_investigation = AsyncMock(return_value=mock_context)
        
        response = test_client.get("/api/v1/investigations/inv_123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["investigation_id"] == "inv_123"
        assert data["status"] == "completed"
    
    def test_get_investigation_not_found(self, test_client, mock_agent):
        """Test getting a non-existent investigation."""
        mock_agent.get_investigation = AsyncMock(return_value=None)
        
        response = test_client.get("/api/v1/investigations/inv_nonexistent")
        
        assert response.status_code == 404
    
    def test_list_investigations(self, test_client, mock_agent):
        """Test listing investigations."""
        from src.agent.core import InvestigationType, TaskPriority
        
        now = datetime.utcnow()
        mock_investigations = [
            MagicMock(
                investigation_id=f"inv_{i}",
                investigation_type=InvestigationType.IOC_ENRICHMENT,
                priority=TaskPriority.MEDIUM,
                status="active",
                created_at=now,
                updated_at=now,
                iocs=[],
                findings=[],
                severity_score=0.5,
            )
            for i in range(3)
        ]
        
        mock_agent.list_investigations = AsyncMock(return_value=mock_investigations)
        
        response = test_client.get("/api/v1/investigations")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["investigations"]) == 3


class TestThreatHuntingEndpoints:
    """Test threat hunting endpoints."""
    
    def test_start_threat_hunt(self, test_client, mock_agent):
        """Test starting a threat hunt."""
        from src.agent.core import InvestigationType, TaskPriority
        
        now = datetime.utcnow()
        mock_context = MagicMock()
        mock_context.investigation_id = "inv_hunt_123"
        mock_context.investigation_type = InvestigationType.THREAT_HUNT
        mock_context.priority = TaskPriority.HIGH
        mock_context.status = "analyzed"
        mock_context.created_at = now
        mock_context.updated_at = now
        mock_context.iocs = []
        mock_context.findings = [{"type": "memory_correlation", "similarity": 0.8}]
        mock_context.severity_score = 0.6
        mock_context.confidence_score = 0.75
        
        mock_agent.hunt_threats = AsyncMock(return_value=mock_context)
        
        response = test_client.post(
            "/api/v1/hunt",
            json={
                "hypothesis": "Possible C2 beacon activity detected",
                "time_range_days": 7
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["investigation_id"] == "inv_hunt_123"
        assert data["investigation_type"] == "threat_hunt"


class TestAttackChainEndpoints:
    """Test attack chain analysis endpoints."""
    
    def test_analyze_attack_chain(self, test_client, mock_agent):
        """Test analyzing an attack chain."""
        mock_analysis = {
            "string_id": "str_123",
            "name": "APT Campaign",
            "confidence": 0.85,
            "total_indicators": 10,
            "timeline": [],
            "techniques": [{"id": "T1071"}],
            "assessment": {"sophistication": "HIGH", "risk_level": "CRITICAL"}
        }
        
        mock_agent.analyze_attack_chain = AsyncMock(return_value=mock_analysis)
        
        response = test_client.post(
            "/api/v1/attack-chains/analyze",
            json={"string_id": "str_123"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["string_id"] == "str_123"
        assert data["assessment"]["sophistication"] == "HIGH"
    
    def test_analyze_attack_chain_not_found(self, test_client, mock_agent):
        """Test analyzing a non-existent attack chain."""
        mock_agent.analyze_attack_chain = AsyncMock(
            return_value={"error": "Attack chain not found"}
        )
        
        response = test_client.post(
            "/api/v1/attack-chains/analyze",
            json={"string_id": "str_nonexistent"}
        )
        
        assert response.status_code == 404
    
    def test_list_attack_chains(self, test_client, mock_agent):
        """Test listing attack chains."""
        now = datetime.utcnow()
        mock_strings = [
            MagicMock(
                id=f"str_{i}",
                name=f"Campaign {i}",
                confidence=0.8,
                beads=[],
                links=[],
                created_at=now,
                updated_at=now,
            )
            for i in range(2)
        ]
        mock_agent.memory_system.bead_memory.strings = {
            s.id: s for s in mock_strings
        }
        
        response = test_client.get("/api/v1/attack-chains")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
    
    def test_get_attack_chain_graph(self, test_client, mock_agent):
        """Test getting attack chain graph."""
        mock_graph = {
            "name": "Campaign Alpha",
            "nodes": [{"id": "n1", "type": "ioc"}],
            "edges": [{"source": "n1", "target": "n2"}],
            "confidence": 0.9
        }
        mock_agent.memory_system.bead_memory.get_string_graph = MagicMock(
            return_value=mock_graph
        )
        
        response = test_client.get("/api/v1/attack-chains/str_123/graph")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Campaign Alpha"
    
    def test_get_attack_chain_graph_not_found(self, test_client, mock_agent):
        """Test getting non-existent attack chain graph."""
        mock_agent.memory_system.bead_memory.get_string_graph = MagicMock(
            return_value=None
        )
        
        response = test_client.get("/api/v1/attack-chains/str_nonexistent/graph")
        
        assert response.status_code == 404


class TestMemoryEndpoints:
    """Test memory query endpoints."""
    
    def test_query_memory(self, test_client, mock_agent):
        """Test querying agent memory."""
        mock_results = {
            "working": [],
            "episodic": [(MagicMock(
                id="mem_1",
                content="Previous investigation about C2",
                metadata={"type": "investigation"},
                timestamp=datetime.utcnow(),
            ), 0.85)],
            "semantic": [],
        }
        mock_agent.memory_system.recall_relevant_context = AsyncMock(
            return_value=mock_results
        )
        
        response = test_client.post(
            "/api/v1/memory/query",
            json={
                "query": "C2 beacon activity",
                "memory_types": ["working", "episodic", "semantic"],
                "top_k": 10
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data["query"] == "C2 beacon activity"
    
    def test_get_memory_stats(self, test_client, mock_agent):
        """Test getting memory statistics."""
        mock_agent.memory_system.get_memory_statistics = MagicMock(return_value={
            "working_memory_entries": 10,
            "episodic_memory_entries": 50,
            "semantic_memory_entries": 100,
            "bead_count": 25,
            "string_count": 5,
        })
        
        response = test_client.get("/api/v1/memory/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "working_memory_entries" in data


class TestMITREEndpoints:
    """Test MITRE ATT&CK mapping endpoints."""
    
    def test_map_to_mitre(self, test_client, mock_agent):
        """Test mapping text to MITRE techniques."""
        response = test_client.get(
            "/api/v1/mitre/map",
            params={
                "text": "Attacker used PowerShell to execute commands",
                "threshold": 0.5
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "techniques" in data
        assert len(data["techniques"]) > 0


class TestRequestValidation:
    """Test request validation."""
    
    def test_enrich_missing_ioc_type(self, test_client):
        """Test enrichment with missing ioc_type."""
        response = test_client.post(
            "/api/v1/enrich",
            json={"ioc_value": "8.8.8.8"}
        )
        
        assert response.status_code == 422
    
    def test_enrich_missing_ioc_value(self, test_client):
        """Test enrichment with missing ioc_value."""
        response = test_client.post(
            "/api/v1/enrich",
            json={"ioc_type": "ipv4"}
        )
        
        assert response.status_code == 422
    
    def test_batch_too_many_iocs(self, test_client):
        """Test batch enrichment with too many IOCs."""
        response = test_client.post(
            "/api/v1/enrich/batch",
            json={
                "iocs": [
                    {"ioc_type": "ipv4", "ioc_value": f"1.2.3.{i}"}
                    for i in range(101)  # Exceeds max of 100
                ]
            }
        )
        
        assert response.status_code == 422
    
    def test_memory_query_invalid_top_k(self, test_client, mock_agent):
        """Test memory query with invalid top_k."""
        response = test_client.post(
            "/api/v1/memory/query",
            json={
                "query": "test",
                "top_k": 500  # Exceeds max of 100
            }
        )
        
        assert response.status_code == 422
