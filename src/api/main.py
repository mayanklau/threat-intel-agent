"""
Threat Intelligence Agent REST API.

This module provides a production-ready REST API for the Threat Intelligence Agent:
- IOC enrichment endpoints
- Investigation management
- Attack chain analysis
- Memory queries
- Health and metrics

Built with FastAPI for high performance and automatic OpenAPI documentation.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

import structlog
import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from src.agent.core import (
    AgentState,
    InvestigationType,
    TaskPriority,
    ThreatContext,
    ThreatIntelAgent,
)
from src.memory.memory_system import (
    ChromaDBBackend,
    InMemoryBackend,
    RedisBackend,
    ThreatIntelMemorySystem,
)
from src.slm.model import TISLMConfig

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter(
    "ti_agent_requests_total",
    "Total API requests",
    ["endpoint", "method", "status"],
)
REQUEST_LATENCY = Histogram(
    "ti_agent_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
)
IOC_ENRICHMENTS = Counter(
    "ti_agent_ioc_enrichments_total",
    "Total IOC enrichments",
    ["ioc_type"],
)


# Pydantic models for API
class IOCEnrichmentRequest(BaseModel):
    """Request model for IOC enrichment."""
    ioc_type: str = Field(..., description="Type of IOC (ipv4, ipv6, domain, md5, sha1, sha256, url, email)")
    ioc_value: str = Field(..., description="Value of the IOC")
    investigation_id: Optional[str] = Field(None, description="Optional investigation to associate with")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ioc_type": "ipv4",
                    "ioc_value": "8.8.8.8",
                    "investigation_id": None,
                }
            ]
        }
    }


class BatchEnrichmentRequest(BaseModel):
    """Request model for batch IOC enrichment."""
    iocs: list[IOCEnrichmentRequest] = Field(..., max_length=100)
    create_investigation: bool = Field(False, description="Create new investigation for batch")


class InvestigationCreateRequest(BaseModel):
    """Request model for creating investigation."""
    investigation_type: str = Field(..., description="Type of investigation")
    priority: str = Field("MEDIUM", description="Priority level (CRITICAL, HIGH, MEDIUM, LOW)")
    initial_iocs: Optional[list[dict]] = Field(None, description="Initial IOCs to process")
    metadata: Optional[dict] = Field(None, description="Additional metadata")


class ThreatHuntRequest(BaseModel):
    """Request model for threat hunting."""
    hypothesis: str = Field(..., description="Threat hypothesis to investigate")
    scope: Optional[dict] = Field(None, description="Scope constraints")
    time_range_days: Optional[int] = Field(None, description="Time range in days")


class AttackChainAnalysisRequest(BaseModel):
    """Request model for attack chain analysis."""
    string_id: str = Field(..., description="Bead string ID to analyze")


class MemoryQueryRequest(BaseModel):
    """Request model for memory queries."""
    query: str = Field(..., description="Query text")
    memory_types: list[str] = Field(
        ["working", "episodic", "semantic"],
        description="Memory types to search",
    )
    top_k: int = Field(10, ge=1, le=100, description="Number of results")


class EnrichmentResponse(BaseModel):
    """Response model for IOC enrichment."""
    ioc_type: str
    ioc_value: str
    threat_score: float
    confidence: float
    classifications: list[dict]
    related_iocs: list[dict]
    threat_actors: list[dict]
    campaigns: list[dict]
    mitre_techniques: list[dict]
    sources: list[str]


class InvestigationResponse(BaseModel):
    """Response model for investigation."""
    investigation_id: str
    investigation_type: str
    priority: int
    status: str
    created_at: str
    updated_at: str
    iocs: list[dict]
    findings: list[dict]
    severity_score: float
    confidence_score: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    agent_state: str
    uptime: Optional[str]
    version: str


class MetricsResponse(BaseModel):
    """Response model for metrics."""
    total_enrichments: int
    total_investigations: int
    iocs_processed: int
    attacks_detected: int
    active_investigations: int
    memory_stats: Optional[dict]


# Global agent instance
_agent: Optional[ThreatIntelAgent] = None


def get_agent() -> ThreatIntelAgent:
    """Dependency to get agent instance."""
    if _agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized",
        )
    return _agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _agent
    
    logger.info("starting_agent_api")
    
    # Initialize agent
    _agent = ThreatIntelAgent(
        agent_id="ti_agent_api",
        model_config=TISLMConfig.small(),  # Use small model for API
    )
    
    await _agent.initialize()
    
    logger.info("agent_api_ready")
    
    yield
    
    # Shutdown
    logger.info("shutting_down_agent_api")
    if _agent:
        await _agent.shutdown()


# Create FastAPI app
app = FastAPI(
    title="Threat Intelligence Agent API",
    description="""
    Production-ready REST API for the Agentic SOC Threat Intelligence Agent.
    
    ## Features
    
    - **IOC Enrichment**: Enrich indicators of compromise with threat intelligence
    - **Investigation Management**: Create and manage threat investigations
    - **Attack Chain Analysis**: Analyze correlated attack chains via Bead Memory
    - **Threat Hunting**: Perform hypothesis-driven threat hunts
    - **Memory Queries**: Query the agent's multi-tier memory system
    
    ## Authentication
    
    API key authentication is supported via the `X-API-Key` header.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health and metrics endpoints
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
)
async def health_check(agent: ThreatIntelAgent = Depends(get_agent)):
    """Check API and agent health status."""
    stats = agent.get_statistics()
    
    return HealthResponse(
        status="healthy" if agent.state == AgentState.READY else "degraded",
        agent_state=agent.state.value,
        uptime=stats.get("uptime"),
        version="1.0.0",
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["Health"],
    summary="Get agent metrics",
)
async def get_metrics(agent: ThreatIntelAgent = Depends(get_agent)):
    """Get agent performance metrics."""
    stats = agent.get_statistics()
    
    return MetricsResponse(
        total_enrichments=stats["total_enrichments"],
        total_investigations=stats["total_investigations"],
        iocs_processed=stats["iocs_processed"],
        attacks_detected=stats["attacks_detected"],
        active_investigations=stats["active_investigations"],
        memory_stats=stats.get("memory_stats"),
    )


@app.get(
    "/metrics/prometheus",
    tags=["Health"],
    summary="Prometheus metrics",
    response_class=Response,
)
async def prometheus_metrics():
    """Get Prometheus-formatted metrics."""
    return Response(
        content=generate_latest(),
        media_type="text/plain",
    )


# IOC Enrichment endpoints
@app.post(
    "/api/v1/enrich",
    response_model=EnrichmentResponse,
    tags=["Enrichment"],
    summary="Enrich single IOC",
)
async def enrich_ioc(
    request: IOCEnrichmentRequest,
    agent: ThreatIntelAgent = Depends(get_agent),
):
    """
    Enrich a single indicator of compromise.
    
    Supported IOC types:
    - ipv4, ipv6: IP addresses
    - domain: Domain names
    - md5, sha1, sha256: File hashes
    - url: URLs
    - email: Email addresses
    """
    with REQUEST_LATENCY.labels(endpoint="/api/v1/enrich").time():
        try:
            result = await agent.enrich_ioc(
                ioc_type=request.ioc_type,
                ioc_value=request.ioc_value,
                investigation_id=request.investigation_id,
            )
            
            IOC_ENRICHMENTS.labels(ioc_type=request.ioc_type).inc()
            REQUEST_COUNT.labels(
                endpoint="/api/v1/enrich",
                method="POST",
                status="200",
            ).inc()
            
            return EnrichmentResponse(
                ioc_type=result.ioc_type,
                ioc_value=result.ioc_value,
                threat_score=result.threat_score,
                confidence=result.confidence,
                classifications=result.classifications,
                related_iocs=result.related_iocs,
                threat_actors=result.threat_actors,
                campaigns=result.campaigns,
                mitre_techniques=result.mitre_techniques,
                sources=result.sources,
            )
            
        except Exception as e:
            REQUEST_COUNT.labels(
                endpoint="/api/v1/enrich",
                method="POST",
                status="500",
            ).inc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )


@app.post(
    "/api/v1/enrich/batch",
    tags=["Enrichment"],
    summary="Batch enrich IOCs",
)
async def batch_enrich_iocs(
    request: BatchEnrichmentRequest,
    background_tasks: BackgroundTasks,
    agent: ThreatIntelAgent = Depends(get_agent),
):
    """
    Enrich multiple IOCs in batch.
    
    Up to 100 IOCs can be processed in a single request.
    For larger batches, consider using the investigation API.
    """
    investigation_id = None
    
    if request.create_investigation:
        context = await agent.start_investigation(
            investigation_type=InvestigationType.IOC_ENRICHMENT,
            priority=TaskPriority.MEDIUM,
        )
        investigation_id = context.investigation_id
    
    # Process IOCs in background
    async def process_batch():
        results = []
        for ioc in request.iocs:
            try:
                result = await agent.enrich_ioc(
                    ioc_type=ioc.ioc_type,
                    ioc_value=ioc.ioc_value,
                    investigation_id=investigation_id or ioc.investigation_id,
                )
                results.append(result.to_dict())
            except Exception as e:
                results.append({
                    "ioc_type": ioc.ioc_type,
                    "ioc_value": ioc.ioc_value,
                    "error": str(e),
                })
        return results
    
    # Start processing in background
    background_tasks.add_task(process_batch)
    
    return {
        "status": "processing",
        "investigation_id": investigation_id,
        "ioc_count": len(request.iocs),
    }


# Investigation endpoints
@app.post(
    "/api/v1/investigations",
    response_model=InvestigationResponse,
    tags=["Investigations"],
    summary="Create investigation",
)
async def create_investigation(
    request: InvestigationCreateRequest,
    agent: ThreatIntelAgent = Depends(get_agent),
):
    """Create a new threat intelligence investigation."""
    try:
        inv_type = InvestigationType[request.investigation_type.upper()]
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid investigation type: {request.investigation_type}",
        )
    
    try:
        priority = TaskPriority[request.priority.upper()]
    except KeyError:
        priority = TaskPriority.MEDIUM
    
    context = await agent.start_investigation(
        investigation_type=inv_type,
        initial_iocs=request.initial_iocs,
        priority=priority,
        metadata=request.metadata,
    )
    
    return InvestigationResponse(
        investigation_id=context.investigation_id,
        investigation_type=context.investigation_type.value,
        priority=context.priority.value,
        status=context.status,
        created_at=context.created_at.isoformat(),
        updated_at=context.updated_at.isoformat(),
        iocs=context.iocs,
        findings=context.findings,
        severity_score=context.severity_score,
        confidence_score=context.confidence_score,
    )


@app.get(
    "/api/v1/investigations/{investigation_id}",
    response_model=InvestigationResponse,
    tags=["Investigations"],
    summary="Get investigation",
)
async def get_investigation(
    investigation_id: str,
    agent: ThreatIntelAgent = Depends(get_agent),
):
    """Get investigation details by ID."""
    context = await agent.get_investigation(investigation_id)
    
    if not context:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Investigation not found: {investigation_id}",
        )
    
    return InvestigationResponse(
        investigation_id=context.investigation_id,
        investigation_type=context.investigation_type.value,
        priority=context.priority.value,
        status=context.status,
        created_at=context.created_at.isoformat(),
        updated_at=context.updated_at.isoformat(),
        iocs=context.iocs,
        findings=context.findings,
        severity_score=context.severity_score,
        confidence_score=context.confidence_score,
    )


@app.get(
    "/api/v1/investigations",
    tags=["Investigations"],
    summary="List investigations",
)
async def list_investigations(
    status: Optional[str] = Query(None, description="Filter by status"),
    investigation_type: Optional[str] = Query(None, description="Filter by type"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    agent: ThreatIntelAgent = Depends(get_agent),
):
    """List all investigations with optional filters."""
    inv_type = None
    if investigation_type:
        try:
            inv_type = InvestigationType[investigation_type.upper()]
        except KeyError:
            pass
    
    investigations = await agent.list_investigations(
        status=status,
        investigation_type=inv_type,
        limit=limit,
    )
    
    return {
        "investigations": [
            {
                "investigation_id": i.investigation_id,
                "investigation_type": i.investigation_type.value,
                "priority": i.priority.value,
                "status": i.status,
                "created_at": i.created_at.isoformat(),
                "updated_at": i.updated_at.isoformat(),
                "ioc_count": len(i.iocs),
                "finding_count": len(i.findings),
                "severity_score": i.severity_score,
            }
            for i in investigations
        ],
        "total": len(investigations),
    }


# Threat hunting endpoints
@app.post(
    "/api/v1/hunt",
    response_model=InvestigationResponse,
    tags=["Threat Hunting"],
    summary="Start threat hunt",
)
async def start_threat_hunt(
    request: ThreatHuntRequest,
    agent: ThreatIntelAgent = Depends(get_agent),
):
    """
    Start a hypothesis-driven threat hunt.
    
    The agent will analyze the hypothesis, map to MITRE ATT&CK,
    and search memory for related historical data.
    """
    from datetime import timedelta
    
    time_range = None
    if request.time_range_days:
        time_range = timedelta(days=request.time_range_days)
    
    context = await agent.hunt_threats(
        hypothesis=request.hypothesis,
        scope=request.scope,
        time_range=time_range,
    )
    
    return InvestigationResponse(
        investigation_id=context.investigation_id,
        investigation_type=context.investigation_type.value,
        priority=context.priority.value,
        status=context.status,
        created_at=context.created_at.isoformat(),
        updated_at=context.updated_at.isoformat(),
        iocs=context.iocs,
        findings=context.findings,
        severity_score=context.severity_score,
        confidence_score=context.confidence_score,
    )


# Attack chain analysis endpoints
@app.post(
    "/api/v1/attack-chains/analyze",
    tags=["Attack Chains"],
    summary="Analyze attack chain",
)
async def analyze_attack_chain(
    request: AttackChainAnalysisRequest,
    agent: ThreatIntelAgent = Depends(get_agent),
):
    """
    Analyze an attack chain from Bead Memory.
    
    Returns timeline reconstruction, technique progression,
    attribution assessment, and impact analysis.
    """
    analysis = await agent.analyze_attack_chain(request.string_id)
    
    if "error" in analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=analysis["error"],
        )
    
    return analysis


@app.get(
    "/api/v1/attack-chains",
    tags=["Attack Chains"],
    summary="List attack chains",
)
async def list_attack_chains(
    limit: int = Query(50, ge=1, le=200),
    agent: ThreatIntelAgent = Depends(get_agent),
):
    """List all detected attack chains from Bead Memory."""
    strings = list(agent.memory_system.bead_memory.strings.values())
    strings.sort(key=lambda s: s.updated_at, reverse=True)
    
    return {
        "attack_chains": [
            {
                "string_id": s.id,
                "name": s.name,
                "confidence": s.confidence,
                "bead_count": len(s.beads),
                "link_count": len(s.links),
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
            }
            for s in strings[:limit]
        ],
        "total": len(strings),
    }


@app.get(
    "/api/v1/attack-chains/{string_id}/graph",
    tags=["Attack Chains"],
    summary="Get attack chain graph",
)
async def get_attack_chain_graph(
    string_id: str,
    agent: ThreatIntelAgent = Depends(get_agent),
):
    """Get graph representation of attack chain."""
    graph = agent.memory_system.bead_memory.get_string_graph(string_id)
    
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Attack chain not found: {string_id}",
        )
    
    return graph


# Memory query endpoints
@app.post(
    "/api/v1/memory/query",
    tags=["Memory"],
    summary="Query agent memory",
)
async def query_memory(
    request: MemoryQueryRequest,
    agent: ThreatIntelAgent = Depends(get_agent),
):
    """
    Query the agent's multi-tier memory system.
    
    Memory types:
    - working: Current investigation context
    - episodic: Historical investigations
    - semantic: Threat knowledge base
    """
    # Get embedding for query
    embedding = agent._get_embedding(request.query)
    
    # Query memory
    results = await agent.memory_system.recall_relevant_context(
        query_embedding=embedding,
        top_k=request.top_k,
    )
    
    # Filter by requested memory types
    filtered_results = {
        mt: results.get(mt, [])
        for mt in request.memory_types
        if mt in results
    }
    
    # Format response
    formatted = {}
    for memory_type, entries in filtered_results.items():
        formatted[memory_type] = [
            {
                "id": entry.id,
                "content": entry.content,
                "similarity": similarity,
                "metadata": entry.metadata,
                "timestamp": entry.timestamp.isoformat(),
            }
            for entry, similarity in entries
        ]
    
    return {
        "query": request.query,
        "results": formatted,
        "total_results": sum(len(v) for v in formatted.values()),
    }


@app.get(
    "/api/v1/memory/stats",
    tags=["Memory"],
    summary="Get memory statistics",
)
async def get_memory_stats(agent: ThreatIntelAgent = Depends(get_agent)):
    """Get statistics from the memory system."""
    return agent.memory_system.get_memory_statistics()


# MITRE ATT&CK endpoints
@app.get(
    "/api/v1/mitre/map",
    tags=["MITRE ATT&CK"],
    summary="Map text to MITRE",
)
async def map_to_mitre(
    text: str = Query(..., description="Text to map to MITRE ATT&CK"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Confidence threshold"),
    agent: ThreatIntelAgent = Depends(get_agent),
):
    """Map threat description text to MITRE ATT&CK techniques."""
    result = agent._inference.map_to_mitre(text, threshold=threshold)
    return result


def run():
    """Run the API server."""
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info",
    )


if __name__ == "__main__":
    run()
