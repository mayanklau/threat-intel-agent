"""
Threat Intelligence Agent Core.

The main orchestration layer that coordinates:
- IOC enrichment through multiple providers
- Threat hunting with hypothesis-driven workflows
- Attack chain analysis and attribution
- Campaign tracking and correlation
- Integration with the multi-tier memory system
"""

from .core import (
    ThreatIntelAgent,
    ThreatContext,
    EnrichmentResult,
    InvestigationType,
    TaskPriority,
    AgentState,
)

__all__ = [
    "ThreatIntelAgent",
    "ThreatContext",
    "EnrichmentResult",
    "InvestigationType",
    "TaskPriority",
    "AgentState",
]
