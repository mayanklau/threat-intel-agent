"""
FastAPI REST API for Threat Intelligence Agent.

Endpoints:
- Health & Metrics: /health, /metrics, /metrics/prometheus
- IOC Enrichment: POST /api/v1/enrich, /api/v1/enrich/batch
- Investigations: CRUD operations for investigations
- Threat Hunting: POST /api/v1/hunt
- Attack Chains: Analysis and visualization endpoints
- Memory: Query and stats for multi-tier memory
- MITRE ATT&CK: Technique mapping

Features:
- OpenAPI documentation at /docs and /redoc
- Prometheus metrics integration
- CORS support for web clients
- Async processing for high concurrency
"""

from .main import app, run

__all__ = ["app", "run"]
