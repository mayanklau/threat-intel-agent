"""
Threat Intelligence Agent for Agentic SOC
==========================================

A production-ready threat intelligence agent featuring:
- Custom Small Language Model (SLM) optimized for security
- Multi-tier memory system with Bead Memory for attack chain correlation
- Integration with major threat intelligence feeds
- RESTful API with async processing

Modules:
    slm: Custom security-focused language model and tokenizer
    memory: Multi-tier memory system (Working, Episodic, Semantic, Bead)
    agent: Core threat intelligence agent orchestration
    integrations: Threat feed providers (OTX, VirusTotal, MISP, etc.)
    api: FastAPI REST endpoints
"""

__version__ = "1.0.0"
__author__ = "Agentic SOC Team"
