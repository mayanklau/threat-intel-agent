"""
Threat Intelligence Agent Core Module.

This module implements the main Threat Intelligence Agent responsible for:
- IOC enrichment and correlation
- Threat feed aggregation
- Attack pattern recognition
- Threat actor attribution
- Campaign tracking
- Risk scoring

The agent uses a custom SLM for inference and a multi-tier memory system
for context management and attack chain correlation.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional
from uuid import uuid4

import numpy as np
import structlog

from src.memory.memory_system import (
    Bead,
    BeadMemory,
    BeadString,
    BeadType,
    MemoryEntry,
    ThreatIntelMemorySystem,
)
from src.slm.model import TaskType, ThreatIntelInference, ThreatIntelSLM, TISLMConfig
from src.slm.tokenizer import SecurityTokenizer

logger = structlog.get_logger()


class AgentState(Enum):
    """Agent lifecycle states."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class InvestigationType(Enum):
    """Types of threat intelligence investigations."""
    IOC_ENRICHMENT = "ioc_enrichment"
    THREAT_HUNT = "threat_hunt"
    INCIDENT_ANALYSIS = "incident_analysis"
    CAMPAIGN_TRACKING = "campaign_tracking"
    ATTRIBUTION = "attribution"
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"


@dataclass
class ThreatContext:
    """Context for threat intelligence operations."""
    investigation_id: str
    investigation_type: InvestigationType
    priority: TaskPriority
    created_at: datetime
    updated_at: datetime
    status: str = "active"
    iocs: list[dict] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)
    related_campaigns: list[str] = field(default_factory=list)
    related_actors: list[str] = field(default_factory=list)
    mitre_mappings: list[dict] = field(default_factory=list)
    severity_score: float = 0.0
    confidence_score: float = 0.0
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "investigation_id": self.investigation_id,
            "investigation_type": self.investigation_type.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status,
            "iocs": self.iocs,
            "findings": self.findings,
            "related_campaigns": self.related_campaigns,
            "related_actors": self.related_actors,
            "mitre_mappings": self.mitre_mappings,
            "severity_score": self.severity_score,
            "confidence_score": self.confidence_score,
            "metadata": self.metadata,
        }


@dataclass
class EnrichmentResult:
    """Result of IOC enrichment."""
    ioc_type: str
    ioc_value: str
    threat_score: float
    confidence: float
    classifications: list[dict]
    related_iocs: list[dict]
    threat_actors: list[dict]
    campaigns: list[dict]
    mitre_techniques: list[dict]
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    sources: list[str] = field(default_factory=list)
    raw_data: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "ioc_type": self.ioc_type,
            "ioc_value": self.ioc_value,
            "threat_score": self.threat_score,
            "confidence": self.confidence,
            "classifications": self.classifications,
            "related_iocs": self.related_iocs,
            "threat_actors": self.threat_actors,
            "campaigns": self.campaigns,
            "mitre_techniques": self.mitre_techniques,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "sources": self.sources,
        }


class ThreatIntelProvider(ABC):
    """Abstract base class for threat intelligence providers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass
    
    @abstractmethod
    async def enrich_ip(self, ip: str) -> dict:
        """Enrich IP address."""
        pass
    
    @abstractmethod
    async def enrich_domain(self, domain: str) -> dict:
        """Enrich domain."""
        pass
    
    @abstractmethod
    async def enrich_hash(self, hash_value: str, hash_type: str) -> dict:
        """Enrich file hash."""
        pass
    
    @abstractmethod
    async def search_ioc(self, ioc: str) -> list[dict]:
        """Search for IOC."""
        pass


class ThreatIntelAgent:
    """
    Main Threat Intelligence Agent.
    
    This agent coordinates threat intelligence operations using:
    - Custom SLM for threat analysis
    - Multi-tier memory system
    - Integration with external threat feeds
    - Attack chain correlation via Bead Memory
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        model_config: Optional[TISLMConfig] = None,
        memory_system: Optional[ThreatIntelMemorySystem] = None,
        providers: Optional[list[ThreatIntelProvider]] = None,
        max_concurrent_tasks: int = 10,
    ):
        self.agent_id = agent_id or f"ti_agent_{uuid4().hex[:8]}"
        self.state = AgentState.IDLE
        self.model_config = model_config or TISLMConfig.medium()
        self.memory_system = memory_system
        self.providers = providers or []
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Model components (initialized lazily)
        self._model: Optional[ThreatIntelSLM] = None
        self._tokenizer: Optional[SecurityTokenizer] = None
        self._inference: Optional[ThreatIntelInference] = None
        
        # Task management
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        # Current investigations
        self._investigations: dict[str, ThreatContext] = {}
        
        # Callbacks
        self._on_finding: Optional[Callable] = None
        self._on_alert: Optional[Callable] = None
        
        # Statistics
        self._stats = {
            "total_enrichments": 0,
            "total_investigations": 0,
            "iocs_processed": 0,
            "attacks_detected": 0,
            "start_time": None,
        }
        
        self.logger = logger.bind(agent_id=self.agent_id)
    
    async def initialize(self) -> bool:
        """Initialize the agent."""
        self.state = AgentState.INITIALIZING
        self.logger.info("agent_initializing")
        
        try:
            # Initialize model
            self._model = ThreatIntelSLM(self.model_config)
            self._tokenizer = SecurityTokenizer()
            self._inference = ThreatIntelInference(
                model=self._model,
                tokenizer=self._tokenizer,
            )
            
            # Initialize memory system if not provided
            if self.memory_system is None:
                from src.memory.memory_system import InMemoryBackend
                self.memory_system = ThreatIntelMemorySystem(
                    working_memory=InMemoryBackend(),
                    episodic_memory=InMemoryBackend(),
                    semantic_memory=InMemoryBackend(),
                )
            
            self.state = AgentState.READY
            self._stats["start_time"] = datetime.utcnow()
            
            self.logger.info(
                "agent_initialized",
                model_params=self._model.num_parameters(),
            )
            
            return True
            
        except Exception as e:
            self.state = AgentState.ERROR
            self.logger.error("agent_initialization_failed", error=str(e))
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the agent gracefully."""
        self.logger.info("agent_shutting_down")
        self.state = AgentState.SHUTDOWN
        
        # Cancel active tasks
        for task_id, task in self._active_tasks.items():
            task.cancel()
            self.logger.debug("task_cancelled", task_id=task_id)
        
        # Wait for tasks to complete
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks.values(), return_exceptions=True)
        
        self.logger.info("agent_shutdown_complete")
    
    def register_finding_callback(self, callback: Callable) -> None:
        """Register callback for new findings."""
        self._on_finding = callback
    
    def register_alert_callback(self, callback: Callable) -> None:
        """Register callback for alerts."""
        self._on_alert = callback
    
    async def _emit_finding(self, finding: dict) -> None:
        """Emit a finding to registered callbacks."""
        if self._on_finding:
            await self._on_finding(finding)
    
    async def _emit_alert(self, alert: dict) -> None:
        """Emit an alert to registered callbacks."""
        if self._on_alert:
            await self._on_alert(alert)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using the SLM."""
        return self._inference.get_embedding(text).cpu().numpy()
    
    async def start_investigation(
        self,
        investigation_type: InvestigationType,
        initial_iocs: Optional[list[dict]] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        metadata: Optional[dict] = None,
    ) -> ThreatContext:
        """Start a new threat intelligence investigation."""
        investigation_id = f"inv_{uuid4().hex[:12]}"
        now = datetime.utcnow()
        
        context = ThreatContext(
            investigation_id=investigation_id,
            investigation_type=investigation_type,
            priority=priority,
            created_at=now,
            updated_at=now,
            iocs=initial_iocs or [],
            metadata=metadata or {},
        )
        
        self._investigations[investigation_id] = context
        self._stats["total_investigations"] += 1
        
        self.logger.info(
            "investigation_started",
            investigation_id=investigation_id,
            investigation_type=investigation_type.value,
            num_iocs=len(initial_iocs) if initial_iocs else 0,
        )
        
        # Process initial IOCs if provided
        if initial_iocs:
            for ioc in initial_iocs:
                await self.enrich_ioc(
                    ioc_type=ioc.get("type"),
                    ioc_value=ioc.get("value"),
                    investigation_id=investigation_id,
                )
        
        return context
    
    async def enrich_ioc(
        self,
        ioc_type: str,
        ioc_value: str,
        investigation_id: Optional[str] = None,
    ) -> EnrichmentResult:
        """
        Enrich an IOC with threat intelligence.
        
        Steps:
        1. Extract features using tokenizer
        2. Get classifications from SLM
        3. Query external providers
        4. Correlate with memory system
        5. Update attack chains
        """
        self.state = AgentState.PROCESSING
        self._stats["iocs_processed"] += 1
        
        self.logger.info(
            "enriching_ioc",
            ioc_type=ioc_type,
            ioc_value=ioc_value[:50],
        )
        
        try:
            # Create description for SLM
            description = f"IOC Type: {ioc_type}\nValue: {ioc_value}"
            
            # Get threat classification
            classification_result = self._inference.classify_threat(description)
            
            # Get severity score
            severity_result = self._inference.score_severity(description)
            
            # Get MITRE mapping
            mitre_result = self._inference.map_to_mitre(description)
            
            # Get embedding for correlation
            embedding = self._get_embedding(description)
            
            # Query external providers
            external_data = await self._query_providers(ioc_type, ioc_value)
            
            # Process through memory system
            bead, string = await self.memory_system.process_ioc(
                ioc_type=ioc_type,
                ioc_value=ioc_value,
                embedding=embedding,
                confidence=severity_result["confidence"],
                source_id=investigation_id,
            )
            
            # Build enrichment result
            result = EnrichmentResult(
                ioc_type=ioc_type,
                ioc_value=ioc_value,
                threat_score=self._calculate_threat_score(
                    severity_result,
                    external_data,
                ),
                confidence=severity_result["confidence"],
                classifications=classification_result["predictions"],
                related_iocs=await self._find_related_iocs(embedding),
                threat_actors=self._extract_actors(external_data),
                campaigns=self._extract_campaigns(external_data, string),
                mitre_techniques=mitre_result["techniques"][:10],
                sources=[p.name for p in self.providers],
                raw_data=external_data,
            )
            
            # Update investigation if provided
            if investigation_id and investigation_id in self._investigations:
                context = self._investigations[investigation_id]
                context.iocs.append(result.to_dict())
                context.severity_score = max(context.severity_score, result.threat_score)
                context.mitre_mappings.extend(result.mitre_techniques)
                context.updated_at = datetime.utcnow()
                
                # Add finding
                finding = {
                    "type": "ioc_enrichment",
                    "ioc": ioc_value,
                    "threat_score": result.threat_score,
                    "severity": severity_result["severity"],
                }
                context.findings.append(finding)
                
                await self._emit_finding(finding)
                
                # Check for high-severity alert
                if result.threat_score >= 0.8:
                    await self._emit_alert({
                        "type": "high_threat_ioc",
                        "investigation_id": investigation_id,
                        "ioc": ioc_value,
                        "threat_score": result.threat_score,
                    })
            
            self._stats["total_enrichments"] += 1
            self.state = AgentState.READY
            
            return result
            
        except Exception as e:
            self.logger.error("enrichment_failed", error=str(e), ioc=ioc_value)
            self.state = AgentState.READY
            raise
    
    async def _query_providers(self, ioc_type: str, ioc_value: str) -> dict:
        """Query all providers for IOC data."""
        results = {}
        
        for provider in self.providers:
            try:
                if ioc_type in ["ipv4", "ipv6"]:
                    data = await provider.enrich_ip(ioc_value)
                elif ioc_type == "domain":
                    data = await provider.enrich_domain(ioc_value)
                elif ioc_type in ["md5", "sha1", "sha256"]:
                    data = await provider.enrich_hash(ioc_value, ioc_type)
                else:
                    data = await provider.search_ioc(ioc_value)
                
                results[provider.name] = data
                
            except Exception as e:
                self.logger.warning(
                    "provider_query_failed",
                    provider=provider.name,
                    error=str(e),
                )
        
        return results
    
    def _calculate_threat_score(self, severity_result: dict, external_data: dict) -> float:
        """Calculate overall threat score."""
        base_score = severity_result["all_scores"].get("CRITICAL", 0) * 1.0 + \
                     severity_result["all_scores"].get("HIGH", 0) * 0.75 + \
                     severity_result["all_scores"].get("MEDIUM", 0) * 0.5 + \
                     severity_result["all_scores"].get("LOW", 0) * 0.25
        
        # Boost based on external data
        external_boost = 0.0
        for provider_data in external_data.values():
            if isinstance(provider_data, dict):
                if provider_data.get("malicious", False):
                    external_boost += 0.2
                if provider_data.get("suspicious", False):
                    external_boost += 0.1
        
        return min(1.0, base_score + external_boost)
    
    async def _find_related_iocs(self, embedding: np.ndarray) -> list[dict]:
        """Find related IOCs using memory system."""
        results = await self.memory_system.semantic_memory.search(
            query_embedding=embedding,
            top_k=10,
        )
        
        related = []
        for entry, similarity in results:
            if similarity > 0.7:
                related.append({
                    "id": entry.id,
                    "content": entry.content,
                    "similarity": similarity,
                    "metadata": entry.metadata,
                })
        
        return related
    
    def _extract_actors(self, external_data: dict) -> list[dict]:
        """Extract threat actors from external data."""
        actors = []
        
        for provider_data in external_data.values():
            if isinstance(provider_data, dict):
                if "threat_actors" in provider_data:
                    actors.extend(provider_data["threat_actors"])
                if "attribution" in provider_data:
                    actors.append(provider_data["attribution"])
        
        return actors
    
    def _extract_campaigns(
        self,
        external_data: dict,
        string: Optional[BeadString],
    ) -> list[dict]:
        """Extract campaign information."""
        campaigns = []
        
        # From external data
        for provider_data in external_data.values():
            if isinstance(provider_data, dict) and "campaigns" in provider_data:
                campaigns.extend(provider_data["campaigns"])
        
        # From bead memory
        if string:
            campaigns.append({
                "id": string.id,
                "name": string.name,
                "confidence": string.confidence,
                "beads": len(string.beads),
                "source": "bead_memory",
            })
        
        return campaigns
    
    async def hunt_threats(
        self,
        hypothesis: str,
        scope: Optional[dict] = None,
        time_range: Optional[timedelta] = None,
    ) -> ThreatContext:
        """
        Perform a threat hunting investigation.
        
        Args:
            hypothesis: The threat hypothesis to investigate
            scope: Scope constraints (e.g., specific systems, networks)
            time_range: Time range to investigate
        """
        self.logger.info("starting_threat_hunt", hypothesis=hypothesis[:100])
        
        # Create investigation
        context = await self.start_investigation(
            investigation_type=InvestigationType.THREAT_HUNT,
            priority=TaskPriority.HIGH,
            metadata={
                "hypothesis": hypothesis,
                "scope": scope or {},
                "time_range": str(time_range) if time_range else None,
            },
        )
        
        # Analyze hypothesis with SLM
        classification = self._inference.classify_threat(hypothesis)
        mitre_mapping = self._inference.map_to_mitre(hypothesis)
        
        context.mitre_mappings = mitre_mapping["techniques"]
        
        # Get embedding for hypothesis
        embedding = self._get_embedding(hypothesis)
        
        # Search memory for related context
        memory_results = await self.memory_system.recall_relevant_context(
            query_embedding=embedding,
            top_k=20,
        )
        
        # Analyze related investigations
        for memory_type, entries in memory_results.items():
            for entry, similarity in entries:
                if similarity > 0.6:
                    context.findings.append({
                        "type": "memory_correlation",
                        "source": memory_type,
                        "content": entry.content[:200],
                        "similarity": similarity,
                        "metadata": entry.metadata,
                    })
        
        # Generate recommendations
        context.metadata["recommendations"] = self._generate_hunt_recommendations(
            hypothesis,
            classification,
            mitre_mapping,
        )
        
        context.status = "analyzed"
        context.updated_at = datetime.utcnow()
        
        return context
    
    def _generate_hunt_recommendations(
        self,
        hypothesis: str,
        classification: dict,
        mitre_mapping: dict,
    ) -> list[dict]:
        """Generate threat hunting recommendations."""
        recommendations = []
        
        # Based on MITRE techniques
        for technique in mitre_mapping.get("techniques", [])[:5]:
            recommendations.append({
                "type": "detection",
                "technique_id": technique["id"],
                "confidence": technique["confidence"],
                "description": f"Look for indicators of {technique['id']}",
            })
        
        # Based on classification
        for pred in classification.get("predictions", [])[:3]:
            recommendations.append({
                "type": "investigation",
                "threat_class": pred["class_id"],
                "confidence": pred["confidence"],
                "description": f"Investigate potential threat class {pred['class_id']}",
            })
        
        return recommendations
    
    async def analyze_attack_chain(
        self,
        string_id: str,
    ) -> dict:
        """
        Analyze an attack chain from Bead Memory.
        
        Returns detailed analysis including:
        - Timeline reconstruction
        - Technique progression
        - Attribution assessment
        - Impact analysis
        """
        graph = self.memory_system.bead_memory.get_string_graph(string_id)
        
        if not graph:
            return {"error": "Attack chain not found"}
        
        # Analyze technique progression
        techniques = []
        for node in graph["nodes"]:
            if node["type"] == "technique":
                techniques.append(node)
        
        # Build timeline
        beads = [
            self.memory_system.bead_memory.beads[bid]
            for bid in graph["nodes"]
            if bid in self.memory_system.bead_memory.beads
        ]
        beads.sort(key=lambda b: b.first_seen)
        
        timeline = [
            {
                "timestamp": b.first_seen.isoformat(),
                "type": b.bead_type.value,
                "value": b.value,
                "confidence": b.confidence,
            }
            for b in beads
        ]
        
        # Generate analysis
        analysis = {
            "string_id": string_id,
            "name": graph["name"],
            "confidence": graph["confidence"],
            "total_indicators": len(graph["nodes"]),
            "total_connections": len(graph["edges"]),
            "timeline": timeline,
            "techniques": techniques,
            "graph": graph,
            "assessment": self._assess_attack_chain(graph, timeline),
        }
        
        return analysis
    
    def _assess_attack_chain(self, graph: dict, timeline: list) -> dict:
        """Generate attack chain assessment."""
        return {
            "sophistication": self._calculate_sophistication(graph),
            "duration": self._calculate_duration(timeline),
            "stage": self._determine_attack_stage(graph),
            "risk_level": self._calculate_risk_level(graph),
        }
    
    def _calculate_sophistication(self, graph: dict) -> str:
        """Calculate attack sophistication level."""
        num_techniques = sum(1 for n in graph["nodes"] if n["type"] == "technique")
        num_connections = len(graph["edges"])
        
        if num_techniques >= 5 and num_connections >= 10:
            return "HIGH"
        elif num_techniques >= 3 or num_connections >= 5:
            return "MEDIUM"
        return "LOW"
    
    def _calculate_duration(self, timeline: list) -> Optional[str]:
        """Calculate attack duration."""
        if len(timeline) < 2:
            return None
        
        first = datetime.fromisoformat(timeline[0]["timestamp"])
        last = datetime.fromisoformat(timeline[-1]["timestamp"])
        duration = last - first
        
        return str(duration)
    
    def _determine_attack_stage(self, graph: dict) -> str:
        """Determine current attack stage based on MITRE."""
        stages = {
            "reconnaissance": 0,
            "initial_access": 0,
            "execution": 0,
            "persistence": 0,
            "privilege_escalation": 0,
            "defense_evasion": 0,
            "credential_access": 0,
            "discovery": 0,
            "lateral_movement": 0,
            "collection": 0,
            "exfiltration": 0,
            "impact": 0,
        }
        
        # Count indicators for each stage
        for node in graph["nodes"]:
            node_type = node.get("type", "").lower()
            if node_type in stages:
                stages[node_type] += 1
        
        # Return most advanced stage with indicators
        for stage in reversed(list(stages.keys())):
            if stages[stage] > 0:
                return stage
        
        return "unknown"
    
    def _calculate_risk_level(self, graph: dict) -> str:
        """Calculate overall risk level."""
        confidence = graph.get("confidence", 0)
        num_nodes = len(graph.get("nodes", []))
        
        if confidence >= 0.8 and num_nodes >= 10:
            return "CRITICAL"
        elif confidence >= 0.6 or num_nodes >= 5:
            return "HIGH"
        elif confidence >= 0.4 or num_nodes >= 3:
            return "MEDIUM"
        return "LOW"
    
    async def get_investigation(self, investigation_id: str) -> Optional[ThreatContext]:
        """Get investigation by ID."""
        return self._investigations.get(investigation_id)
    
    async def list_investigations(
        self,
        status: Optional[str] = None,
        investigation_type: Optional[InvestigationType] = None,
        limit: int = 50,
    ) -> list[ThreatContext]:
        """List investigations with optional filters."""
        investigations = list(self._investigations.values())
        
        if status:
            investigations = [i for i in investigations if i.status == status]
        
        if investigation_type:
            investigations = [i for i in investigations if i.investigation_type == investigation_type]
        
        # Sort by updated_at descending
        investigations.sort(key=lambda x: x.updated_at, reverse=True)
        
        return investigations[:limit]
    
    def get_statistics(self) -> dict:
        """Get agent statistics."""
        uptime = None
        if self._stats["start_time"]:
            uptime = str(datetime.utcnow() - self._stats["start_time"])
        
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "uptime": uptime,
            "total_enrichments": self._stats["total_enrichments"],
            "total_investigations": self._stats["total_investigations"],
            "iocs_processed": self._stats["iocs_processed"],
            "attacks_detected": self._stats["attacks_detected"],
            "active_investigations": len([
                i for i in self._investigations.values()
                if i.status == "active"
            ]),
            "memory_stats": self.memory_system.get_memory_statistics()
            if self.memory_system else None,
        }
