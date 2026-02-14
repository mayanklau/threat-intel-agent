"""
Test configuration and fixtures for Threat Intelligence Agent.
"""

import asyncio
from datetime import datetime
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    from src.config import Settings, SLMConfig, MemoryConfig, ThreatFeedConfig, APIConfig, AgentConfig
    
    return Settings(
        environment="development",
        debug=True,
        slm=SLMConfig(
            model_size="small",
            vocab_size=10000,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=512,
        ),
        memory=MemoryConfig(
            redis_url="redis://localhost:6379/0",
            chroma_host="localhost",
            chroma_port=8000,
        ),
        threat_feeds=ThreatFeedConfig(),
        api=APIConfig(port=8081),
        agent=AgentConfig(
            enrichment_providers=["otx"],
            max_concurrent_tasks=2,
        ),
    )


@pytest.fixture
def sample_iocs() -> list[str]:
    """Sample IOCs for testing."""
    return [
        "192.168.1.100",
        "10.0.0.1",
        "evil-domain.com",
        "malware.example.org",
        "d41d8cd98f00b204e9800998ecf8427e",  # MD5
        "da39a3ee5e6b4b0d3255bfef95601890afd80709",  # SHA1
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",  # SHA256
        "CVE-2024-1234",
        "T1059.001",  # MITRE technique
    ]


@pytest.fixture
def sample_threat_context():
    """Sample threat context for testing."""
    return {
        "investigation_id": "inv-12345",
        "iocs": ["192.168.1.100", "evil.com"],
        "findings": [
            {"type": "malware", "name": "Cobalt Strike", "confidence": 0.85}
        ],
        "campaigns": ["APT29-2024"],
        "actors": ["APT29"],
        "mitre_mappings": {
            "T1059.001": "PowerShell",
            "T1053.005": "Scheduled Task"
        },
        "severity": "high",
        "confidence": 0.8,
    }


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock = MagicMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=True)
    mock.expire = AsyncMock(return_value=True)
    mock.keys = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def mock_chromadb():
    """Mock ChromaDB client for testing."""
    mock = MagicMock()
    mock.get_or_create_collection = MagicMock()
    mock.get_collection = MagicMock()
    collection = MagicMock()
    collection.add = MagicMock()
    collection.query = MagicMock(return_value={
        "ids": [["doc1"]],
        "documents": [["Test document"]],
        "distances": [[0.1]],
        "metadatas": [[{"source": "test"}]],
    })
    mock.get_or_create_collection.return_value = collection
    return mock


@pytest.fixture
def mock_otx_response():
    """Mock OTX API response."""
    return {
        "general": {
            "pulse_info": {
                "count": 5,
                "pulses": [
                    {
                        "name": "Test Pulse",
                        "tags": ["malware", "apt"],
                        "created": "2024-01-01T00:00:00Z",
                    }
                ]
            },
            "reputation": 50,
            "whois": {},
        },
        "malware": {
            "data": [
                {"hash": "abc123", "name": "TestMalware"}
            ]
        },
        "passive_dns": {
            "passive_dns": [
                {"hostname": "evil.com", "address": "192.168.1.100"}
            ]
        }
    }


@pytest.fixture
def mock_virustotal_response():
    """Mock VirusTotal API response."""
    return {
        "data": {
            "attributes": {
                "last_analysis_stats": {
                    "malicious": 15,
                    "suspicious": 3,
                    "harmless": 50,
                    "undetected": 12,
                },
                "reputation": -25,
                "last_analysis_results": {
                    "Kaspersky": {"category": "malicious", "result": "Trojan.Generic"},
                    "Symantec": {"category": "malicious", "result": "Trojan.Gen"},
                }
            }
        }
    }


@pytest.fixture
def mock_abuseipdb_response():
    """Mock AbuseIPDB API response."""
    return {
        "data": {
            "ipAddress": "192.168.1.100",
            "isPublic": True,
            "abuseConfidenceScore": 75,
            "countryCode": "US",
            "usageType": "Data Center/Web Hosting",
            "isp": "Test ISP",
            "totalReports": 150,
            "numDistinctUsers": 25,
            "lastReportedAt": "2024-01-15T12:00:00Z",
        }
    }
